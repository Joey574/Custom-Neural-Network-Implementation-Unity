using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UIElements;

public class NeuralNetworkMatrixBased : MonoBehaviour
{
    [Header("Neural Net Size")]
    public int inputSize;
    public int hiddenSize;
    public int outputSize;

    [Header("Neural Net Data")]
    private float [,] W1;
    private float [,] W2;

    private float[] B1;
    private float[] B2;

    private float[] A1;
    private float[] A2;

    void Awake()
    {
        InitializeNetwork();
        TestNetwork();
    }

    private void InitializeNetwork()
    {
        W1 = new float[inputSize, hiddenSize];
        W2 = new float[hiddenSize, outputSize];

        B1 = new float[hiddenSize];
        B2 = new float[outputSize];

        A1 = new float[hiddenSize];
        A2 = new float[outputSize];

        for (int x = 0; x < inputSize; x++)
        {
            for (int y = 0; y < hiddenSize; y++)
            {
                W1[x, y] = Random.Range(-0.5f, 0.5f);
            }
        }

        for (int x = 0; x < hiddenSize; x++)
        {
            for (int y = 0; y < outputSize; y++)
            {
                W2[x, y] = Random.Range(-0.5f, 0.5f);
            }
        }

        for (int i = 0; i < hiddenSize; i++)
        {
            B1[i] = Random.Range(-0.5f, 0.5f);
        }

        for (int i = 0; i < outputSize; i++)
        {
            B2[i] = Random.Range(-0.5f, 0.5f);
        }

        Debug.Log("W1 Size: " + W1.Length);
        Debug.Log("W2 Size: " + W2.Length);

        Debug.Log("B1 Size: " + B1.Length);
        Debug.Log("B2 Size: " + B2.Length);

        Debug.Log("Total Connections: " + (W1.Length + W2.Length + B1.Length + B2.Length));
    }

    private void TestNetwork()
    {
        float[] x = new float[inputSize];

        for (int i = 0; i < inputSize; i++)
        {
            x[i] = Random.Range(-10, 10);
        }

        ForwardProp(x);
        DisplayProbabilities();
    }

    private void DisplayProbabilities()
    {
        for (int i = 0; i < outputSize; i++)
        {
            Debug.Log("Probability of class [" + i + "]: " + A2[i]);
        }

        Debug.Log("Sum of probabilities: " + A2.Sum());
    }

    private void ForwardProp(float[] x)
    {
        for (int i = 0; i < hiddenSize; i++)
        {
            for (int  j = 0; j < inputSize; j++)
            {
                A1[i] += x[j] * W1[j,i];
            }
            A1[i] += B1[i];
            A1[i] = Sigmoid(A1[i]);
        }

        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < hiddenSize; j++)
            {
                A2[i] += A1[j] * W2[j,i];
            }
            A2[i] += B2[i];
        }
        A2 = Softmax(A2);
    }

    private void BackwardProp()
    {

    }

    private void UpdateNetwork()
    {

    }

    private float[] Softmax(float[] A)
    {
        float[] predictions = new float[outputSize];
        float sum = 0;

        for (int i = 0; i < A.Length; i++)
        {
            sum += Mathf.Exp(A[i]);
        }

        for (int i = 0; i < A.Length; i++)
        {
            predictions[i] = Mathf.Exp(A[i]) / sum;
        }

        return predictions;
    }

    private float Sigmoid(float x)
    {
        return 1 / (1 + Mathf.Exp(-x));
    }
}
