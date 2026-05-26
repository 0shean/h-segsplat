using System;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.Rendering;
using GaussianSplatting.Runtime;

/// <summary>
/// Inspector-driven H-SegSplat client. Attach to the Splats GameObject.
///
/// Type into Child / Parent / Grandparent fields below. Empty strings = skip
/// that level. Then right-click the component header in the Inspector and pick
/// "Submit Query" (or "Clear Query") from the context menu.
///
/// Sends the three terms to the local FastAPI server (see
/// second_stage/viewer/server/serve.py), receives a float[G] of per-Gaussian
/// relevancy, writes it into the GaussianSplatRenderer's internal
/// _SplatQuerySimilarities buffer via reflection. The patched render shader
/// reads that buffer and tints relevant splats magenta.
///
/// [ExecuteAlways] so this works in Edit mode — no Play required.
/// </summary>
[ExecuteAlways]
public class HSegSplatClient : MonoBehaviour
{
    [Header("Server")]
    public string serverUrl = "http://127.0.0.1:8000";

    [Header("Splat renderer (auto-found if empty)")]
    public GaussianSplatRenderer splatRenderer;

    [Header("Query terms — leave a field empty to skip that level")]
    public string child = "";
    public string parent = "";
    public string grandparent = "";

    [Header("Levels for each term (v1 only — v2 binds by query shape)")]
    public int childLevel = 6;
    public int parentLevel = 3;
    public int grandparentLevel = 1;

    [Header("Query path")]
    [Tooltip("If true, uses /query_combined_v2 (per-mask + containment + fluid binding). " +
             "Requires gaussians.pt to have the v2 payload (re-run build + Colab inference " +
             "with the updated scripts). If false, uses the v1 endpoint and the per-level fields above.")]
    public bool useV2 = true;

    [Header("Highlight threshold (relevancy >= this => splat is tinted magenta)")]
    [Range(0f, 1f)]
    public float threshold = 0.5f;

    [Header("Last query status (read-only)")]
    [TextArea(3, 6)]
    public string lastStatus = "(no query yet)";

    [Serializable]
    private class QueryBody
    {
        public string child;
        public string parent;
        public string grandparent;
        public int child_level;
        public int parent_level;
        public int grandparent_level;
    }

    // m_GpuQuerySimilarities is declared `internal` in the package, so we reach
    // it via reflection rather than patching the package.
    private FieldInfo m_QuerySimField;

    void OnEnable()
    {
        ResolveRenderer();
    }

    void ResolveRenderer()
    {
        if (splatRenderer == null)
            splatRenderer = GetComponent<GaussianSplatRenderer>();
        if (splatRenderer == null)
            splatRenderer = FindObjectOfType<GaussianSplatRenderer>();
        m_QuerySimField = typeof(GaussianSplatRenderer).GetField(
            "m_GpuQuerySimilarities",
            BindingFlags.NonPublic | BindingFlags.Instance);
    }

    private GraphicsBuffer GetQuerySimBuffer()
    {
        if (m_QuerySimField == null || splatRenderer == null) return null;
        return m_QuerySimField.GetValue(splatRenderer) as GraphicsBuffer;
    }

    // ------------------------------------------------------------------------
    // Inspector context-menu entries: right-click the component header.
    // ------------------------------------------------------------------------

    [ContextMenu("Submit Query")]
    public async void SubmitQuery()
    {
        ResolveRenderer();
        if (splatRenderer == null)
        {
            SetStatus("error: no GaussianSplatRenderer found");
            return;
        }

        bool anyTerm = !string.IsNullOrWhiteSpace(child)
                     || !string.IsNullOrWhiteSpace(parent)
                     || !string.IsNullOrWhiteSpace(grandparent);
        if (!anyTerm)
        {
            SetStatus("nothing to query — all three fields are empty");
            return;
        }

        string endpoint = useV2 ? "/query_combined_v2" : "/query_combined";
        SetStatus($"querying [{(useV2 ? "v2" : "v1")}]: child='{child}' parent='{parent}' grandparent='{grandparent}' ...");

        var body = new QueryBody
        {
            child = child ?? "",
            parent = parent ?? "",
            grandparent = grandparent ?? "",
            child_level = childLevel,
            parent_level = parentLevel,
            grandparent_level = grandparentLevel,
        };
        string json = JsonUtility.ToJson(body);

        float[] relevancy = await PostBinary(serverUrl + endpoint, json);
        if (relevancy == null)
        {
            SetStatus("error: request failed (see Console for details)");
            return;
        }

        if (relevancy.Length != splatRenderer.splatCount)
        {
            SetStatus($"error: received {relevancy.Length} values but splatCount is {splatRenderer.splatCount}");
            return;
        }

        float minVal = float.PositiveInfinity, maxVal = float.NegativeInfinity;
        int nAbove = 0;
        for (int i = 0; i < relevancy.Length; i++)
        {
            float v = relevancy[i];
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
            if (v >= threshold) nAbove++;
        }

        splatRenderer.m_MinRelevancyScore = threshold;
        splatRenderer.m_MaxRelevancyScore = Mathf.Max(maxVal, threshold + 1e-3f);
        Shader.SetGlobalFloat("_MinRelevancyScore", threshold);
        Shader.SetGlobalFloat("_MaxRelevancyScore", splatRenderer.m_MaxRelevancyScore);

        var buf = GetQuerySimBuffer();
        if (buf == null)
        {
            SetStatus("error: m_GpuQuerySimilarities is null (renderer may not have initialized; try entering Play mode once)");
            return;
        }
        buf.SetData(relevancy);

        SetStatus($"done: received {relevancy.Length} values, "
                  + $"min={minVal:F4}, max={maxVal:F4}, "
                  + $"threshold={threshold:F2}, splats highlighted: {nAbove}");

        // Nudge the editor to repaint so the highlight is visible immediately.
#if UNITY_EDITOR
        UnityEditor.SceneView.RepaintAll();
#endif
    }

    [ContextMenu("Clear Query")]
    public void ClearQuery()
    {
        ResolveRenderer();
        var buf = GetQuerySimBuffer();
        if (buf == null)
        {
            SetStatus("nothing to clear (no buffer)");
            return;
        }
        var zeros = new float[splatRenderer.splatCount];
        buf.SetData(zeros);
        Shader.SetGlobalFloat("_MinRelevancyScore", 2.0f); // > 1 → nothing passes
        SetStatus("cleared.");
#if UNITY_EDITOR
        UnityEditor.SceneView.RepaintAll();
#endif
    }

    private void SetStatus(string msg)
    {
        lastStatus = msg;
        Debug.Log("[HSegSplat] " + msg);
    }

    // ------------------------------------------------------------------------
    // Networking
    // ------------------------------------------------------------------------

    private async Task<float[]> PostBinary(string url, string jsonBody)
    {
        using (var request = new UnityWebRequest(url, "POST"))
        {
            byte[] bodyBytes = Encoding.UTF8.GetBytes(jsonBody);
            request.uploadHandler = new UploadHandlerRaw(bodyBytes);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            var op = request.SendWebRequest();
            while (!op.isDone) await Task.Yield();

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"[HSegSplat] {url} failed: {request.error} (code {request.responseCode})");
                if (request.downloadHandler != null)
                    Debug.LogError($"[HSegSplat] response: {request.downloadHandler.text}");
                return null;
            }

            byte[] data = request.downloadHandler.data;
            int n = data.Length / 4;
            var floats = new float[n];
            Buffer.BlockCopy(data, 0, floats, 0, data.Length);
            return floats;
        }
    }
}
