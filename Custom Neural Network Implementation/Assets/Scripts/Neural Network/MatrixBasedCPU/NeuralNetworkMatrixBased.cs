using System.Linq;
using UnityEngine;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using Unity.VisualScripting;

public class NeuralNetworkMatrixBased : MonoBehaviour
{
    [Header("Neural Net Hyper Params")]
    public int inputSize;
    public int hiddenSize;
    public int outputSize;

    public float learningRate;
    public int dataSize;

    [Header("Neural Net Data")]
    private Matrix<float> W1;
    private Matrix<float> W2;

    private Vector<float> B1;
    private Vector<float> B2;

    [Header("Neural Net Outputs")]
    private Matrix<float> A1Total;
    private Matrix<float> A2Total;

    private Matrix<float> A1;
    private Matrix<float> A2;

    [Header("Derivative Values")]
    private Matrix<float> dW1;
    private Matrix<float> dW2;

    private Matrix<float> dZ1;
    private Matrix<float> dZ2;

    private Vector<float> dB1;
    private Vector<float> dB2;    

    void Awake()
    {
        InitializeNetwork();
        TestNetwork();
    }

    private void InitializeNetwork()
    {
        B1 = Vector<float>.Build.Dense(hiddenSize);
        B2 = Vector<float>.Build.Dense(outputSize);

        A1 = Matrix<float>.Build.Dense(hiddenSize, dataSize);
        A1Total = Matrix<float>.Build.Dense(hiddenSize, dataSize);

        A2 = Matrix<float>.Build.Dense(outputSize, dataSize);
        A2Total = Matrix<float>.Build.Dense(outputSize, dataSize);

        W1 = Matrix<float>.Build.Dense(inputSize, hiddenSize);
        W2 = Matrix<float>.Build.Dense(hiddenSize, outputSize);

        dW1 = Matrix<float>.Build.Dense(W1.RowCount, W1.ColumnCount);
        dW2 = Matrix<float>.Build.Dense(W2.RowCount, W2.ColumnCount);

        dZ1 = Matrix<float>.Build.Dense(A1Total.RowCount, A1Total.ColumnCount);
        dZ2 = Matrix<float>.Build.Dense(A2Total.RowCount, A2Total.ColumnCount);

        dB1 = Vector<float>.Build.Dense(B1.Count);
        dB2 = Vector<float>.Build.Dense(B2.Count);

        for (int row = 0; row < inputSize; row++)
        {
            for (int col = 0; col < hiddenSize; col++)
            {
                W1[row, col] = Random.Range(-0.5f, 0.5f);
            }
        }

        for (int row = 0; row < hiddenSize; row++)
        {
            for (int col = 0; col < outputSize; col++)
            {
                W2[row, col] = Random.Range(-0.5f, 0.5f);
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

        Debug.Log("W1 Connections: " + (W1.ColumnCount * W1.RowCount));
        Debug.Log("W2 Connection: " + (W2.ColumnCount * W1.RowCount));

        Debug.Log("B1 Size: " + B1.Count);
        Debug.Log("B2 Size: " + B2.Count);

        Debug.Log("Total Connections: " + ((W1.RowCount * W1.ColumnCount) + (W2.RowCount * W2.ColumnCount) + B1.Count + B2.Count));
        Debug.Log("INITIALIZATION COMPLETE");
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
            Debug.Log("Probability of class [" + i + "]: " + A2.Column(A2.ColumnCount - 1).Sum());
        }

        Debug.Log("Sum of probabilities: " + A2.Column(A2.ColumnCount - 1).Sum());
    }

    private void ForwardProp(float[] x)
    {
        Vector<float> vecX = Vector<float>.Build.Dense(x);

        for (int i = 0; i < hiddenSize; i++)
        {
            A1Total[i] = W1.Column(i).DotProduct(vecX) + B1[i];
            A1[i] = Sigmoid(A1Total[i]);
        }

        for (int i = 0; i < outputSize; i++)
        {
            A2Total[i] = W2.Column(i).DotProduct(A1) + B2[i];
        }
        A2 = Softmax(A2Total);
    }

    private void BackwardProp()
    {
        //dZ2 = A2 - y;


    }

    private void UpdateNetwork()
    {
        W1 = W1 - learningRate * dW1;
        W2 = W2 - learningRate * dW2;

        B1 = B1 - learningRate * dB1;
        B2 = B2 - learningRate * dB2;
    }

    private Vector<float> Softmax(Vector<float> A)
    {
        Vector<float> predictions = Vector<float>.Build.Dense(outputSize); ;
        float sum = 0;

        for (int i = 0; i < A.Count; i++)
        {
            sum += Mathf.Exp(A[i]);
        }

        for (int i = 0; i < A.Count; i++)
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
