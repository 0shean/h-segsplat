#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;
using GaussianSplatting.Runtime;

public static class SetNearCutoff
{
    private static void SetCutoff(float v)
    {
        var r = Object.FindObjectOfType<GaussianSplatRenderer>();
        if (r == null) { Debug.LogError("no renderer"); return; }
        r.m_NearPlaneCutoff = v;
        Debug.Log($"Set m_NearPlaneCutoff to {r.m_NearPlaneCutoff}");
        EditorUtility.SetDirty(r);
        UnityEditor.SceneView.RepaintAll();
    }

    [MenuItem("Tools/HSegSplat/Cutoff 0.5")] public static void C05() => SetCutoff(0.5f);
    [MenuItem("Tools/HSegSplat/Cutoff 1.0")] public static void C10() => SetCutoff(1.0f);
    [MenuItem("Tools/HSegSplat/Cutoff 1.5")] public static void C15() => SetCutoff(1.5f);
    [MenuItem("Tools/HSegSplat/Cutoff 2.0")] public static void C20() => SetCutoff(2.0f);
    [MenuItem("Tools/HSegSplat/Cutoff 3.0")] public static void C30() => SetCutoff(3.0f);
    [MenuItem("Tools/HSegSplat/Cutoff 5.0")] public static void C50() => SetCutoff(5.0f);

    [MenuItem("Tools/HSegSplat/Print Near Cutoff")]
    public static void PrintIt()
    {
        var r = Object.FindObjectOfType<GaussianSplatRenderer>();
        if (r == null) { Debug.LogError("no renderer"); return; }
        Debug.Log($"m_NearPlaneCutoff = {r.m_NearPlaneCutoff}");
    }
}
#endif
