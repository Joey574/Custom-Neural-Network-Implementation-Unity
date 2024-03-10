using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NetworkManager : MonoBehaviour
{
    [Header("Compute Shaders")]
    public ComputeShader ForwardPropogation;
    public ComputeShader BackwardPropogation;
    public ComputeShader UpdateNetwork;

    int fpKernelID;
    int bpKernelID;
    int udnKernelID;

    [Header("Hyperparameters")]
    public int InputSize;
    public int OutputSize;
    public List<int> HiddenSize;
    
    [Header("Training Hyperparameters")]
    public float LearningRate;
    public float thresholdAccuracy;
    public int batchSize;

    [Header("Goofy ahh network")]
    public List<RenderTexture> Weights;
    public List<RenderTexture> Biases;

    public List<RenderTexture> Activation;
    public List<RenderTexture> ATotal;


    void Awake()
    {
        CreateTextures();

        fpKernelID = ForwardPropogation.FindKernel("CSMain");
        bpKernelID = BackwardPropogation.FindKernel("CSMain");
        udnKernelID = UpdateNetwork.FindKernel("CSMain");
    }

    void Update()
    {
        
    }

    void CreateTextures()
    {

    }

    private void DispatchKernals()
    {

    }
}
