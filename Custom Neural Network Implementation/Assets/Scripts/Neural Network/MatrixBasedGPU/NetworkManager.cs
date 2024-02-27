using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NetworkManager : MonoBehaviour
{
    [Header("Compute Shaders")]
    public ComputeShader ForwardPropogation;
    public ComputeShader BackwardPropogation;
    public ComputeShader UpdateNetwork;

    [Header("Hyperperameters")]
    public int InputSize;
    public int OutputSize;
    public List<int> HiddenSize;

    public float LearningRate;

    void Awake()
    {
        
    }

    void Update()
    {
        
    }

    private void DispatchKernals()
    {
    }
}
