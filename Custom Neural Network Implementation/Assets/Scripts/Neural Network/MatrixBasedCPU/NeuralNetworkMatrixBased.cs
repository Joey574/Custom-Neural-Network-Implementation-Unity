using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using Unity.VisualScripting;
using UnityEngine;

public class NeuralNetworkMatrixBased : MonoBehaviour
{
    [Header("Neural Net Size")]
    public int inputSize;
    public int hiddenSize;
    public int outputSize;

    [Header("Neural Net Data")]
    private Vector<Vector<float>> W1;
    private Vector<Vector<float>> W2;

    private float[] B1;
    private float[] B2;

    void Awake()
    {
        InitializeNetwork();
    }

    private void InitializeNetwork()
    {
        W1 = new float[hiddenSize,inputSize];
        W2 = new float[outputSize, hiddenSize];

        B1 = new float[hiddenSize];
        B2 = new float[outputSize];

        W1V = new Vector<float>(10);



        for (int x = 0; x < hiddenSize; x++)
        {
            B1[x] = Random.Range(-0.5f, 0.5f);
            for (int  y = 0; y < inputSize; y++)
            {
                W1[x,y] = Random.Range(-0.5f, 0.5f);
            }
        }

        for (int x = 0; x < outputSize; x++)
        {
            B2[x] = Random.Range(-0.5f, 0.5f);
            for (int y = 0; y < hiddenSize; y++)
            {
                W2[x, y] = Random.Range(-0.5f, 0.5f);
            }
        }

        Debug.Log("W1 Size: " + W1.Length);
        Debug.Log("W2 Size: " + W2.Length);

        Debug.Log("B1 Size: " + B1.Length);
        Debug.Log("B2 Size: " + B2.Length);
    }

    private void ForwardProp(float[] x)
    {
       
    }

    private void BackwardProp()
    {

    }

    private void UpdateNetwork()
    {

    }
}
