#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

/// <summary>
/// Custom Inspector for HSegSplatClient: adds two big buttons below the
/// default fields. Right-click context menu still works too.
/// </summary>
[CustomEditor(typeof(HSegSplatClient))]
public class HSegSplatClientEditor : Editor
{
    public override void OnInspectorGUI()
    {
        // Draw the normal fields first.
        DrawDefaultInspector();

        var client = (HSegSplatClient)target;

        EditorGUILayout.Space(8);

        GUILayout.BeginHorizontal();
        if (GUILayout.Button("Submit Query", GUILayout.Height(32)))
        {
            client.SubmitQuery();
        }
        if (GUILayout.Button("Clear Query", GUILayout.Height(32)))
        {
            client.ClearQuery();
        }
        GUILayout.EndHorizontal();
    }
}
#endif
