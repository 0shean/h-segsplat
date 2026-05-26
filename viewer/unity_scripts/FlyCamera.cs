using UnityEngine;

/// <summary>
/// Minimal WASD + mouse-look fly camera for Play mode.
///
/// Hold right mouse button to look around. WASD to move, Q/E for down/up.
/// Shift for fast. Mouse cursor is captured while RMB is held.
/// </summary>
public class FlyCamera : MonoBehaviour
{
    [Header("Speed")]
    public float baseSpeed = 2.0f;
    public float fastMultiplier = 5.0f;

    [Header("Look")]
    public float mouseSensitivity = 2.0f;

    private float pitch = 0f;   // x rotation
    private float yaw = 0f;     // y rotation
    private bool looking = false;

    void Start()
    {
        var e = transform.eulerAngles;
        pitch = e.x;
        yaw = e.y;
    }

    void Update()
    {
        // Toggle look on right mouse button.
        if (Input.GetMouseButtonDown(1))
        {
            looking = true;
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }
        if (Input.GetMouseButtonUp(1))
        {
            looking = false;
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        }

        if (looking)
        {
            yaw += Input.GetAxis("Mouse X") * mouseSensitivity;
            pitch -= Input.GetAxis("Mouse Y") * mouseSensitivity;
            pitch = Mathf.Clamp(pitch, -89f, 89f);
            transform.eulerAngles = new Vector3(pitch, yaw, 0);
        }

        float speed = baseSpeed * (Input.GetKey(KeyCode.LeftShift) ? fastMultiplier : 1f);
        Vector3 move = Vector3.zero;
        if (Input.GetKey(KeyCode.W)) move += transform.forward;
        if (Input.GetKey(KeyCode.S)) move -= transform.forward;
        if (Input.GetKey(KeyCode.A)) move -= transform.right;
        if (Input.GetKey(KeyCode.D)) move += transform.right;
        if (Input.GetKey(KeyCode.E)) move += transform.up;
        if (Input.GetKey(KeyCode.Q)) move -= transform.up;
        transform.position += move * speed * Time.deltaTime;
    }
}
