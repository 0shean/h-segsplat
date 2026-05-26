using System.Threading.Tasks;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Tiny UI controller for the H-SegSplat viewer.
///
/// Wire in the inspector:
///   - childInput, parentInput, grandparentInput  (TMP_InputField each)
///   - submitButton                                (Button)
///   - clearButton                                 (Button, optional)
///   - statusText                                  (TMP_Text, optional)
///   - client                                      (HSegSplatClient)
///
/// Pressing Enter while focused in any input field also submits.
///
/// [ExecuteAlways] so the UI also reacts in Edit mode — Play mode not needed.
/// </summary>
[ExecuteAlways]
public class HSegSplatUI : MonoBehaviour
{
    [Header("Inputs")]
    public TMP_InputField childInput;
    public TMP_InputField parentInput;
    public TMP_InputField grandparentInput;

    [Header("Buttons")]
    public Button submitButton;
    public Button clearButton;

    [Header("Status")]
    public TMP_Text statusText;

    [Header("Client")]
    public HSegSplatClient client;

    void Start()
    {
        if (submitButton != null) submitButton.onClick.AddListener(OnSubmitClicked);
        if (clearButton != null) clearButton.onClick.AddListener(OnClearClicked);

        if (childInput != null) childInput.onSubmit.AddListener(_ => OnSubmitClicked());
        if (parentInput != null) parentInput.onSubmit.AddListener(_ => OnSubmitClicked());
        if (grandparentInput != null) grandparentInput.onSubmit.AddListener(_ => OnSubmitClicked());

        if (client == null) client = FindObjectOfType<HSegSplatClient>();
        if (client == null) Debug.LogError("[HSegSplatUI] No HSegSplatClient found.");

        SetStatus("Ready.");
    }

    private async void OnSubmitClicked()
    {
        string c = childInput != null ? childInput.text : "";
        string p = parentInput != null ? parentInput.text : "";
        string gp = grandparentInput != null ? grandparentInput.text : "";

        SetStatus($"querying… child='{c}' parent='{p}' grandparent='{gp}'");
        if (submitButton != null) submitButton.interactable = false;

        try
        {
            await client.SubmitQuery(c, p, gp);
            SetStatus($"done: child='{c}' parent='{p}' grandparent='{gp}'");
        }
        catch (System.Exception e)
        {
            SetStatus($"error: {e.Message}");
            Debug.LogException(e);
        }
        finally
        {
            if (submitButton != null) submitButton.interactable = true;
        }
    }

    private void OnClearClicked()
    {
        if (client != null) client.ClearQuery();
        SetStatus("cleared.");
    }

    private void SetStatus(string msg)
    {
        if (statusText != null) statusText.text = msg;
        Debug.Log("[HSegSplatUI] " + msg);
    }
}
