using System.Linq;
using UnityEngine;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using Unity.VisualScripting;
using System;

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
        // Vectors
        B1 = Vector<float>.Build.Dense(hiddenSize);
        B2 = Vector<float>.Build.Dense(outputSize);

        dB1 = Vector<float>.Build.Dense(B1.Count);
        dB2 = Vector<float>.Build.Dense(B2.Count);

        // Matrices
        A1 = Matrix<float>.Build.Dense(hiddenSize, dataSize);
        A1Total = Matrix<float>.Build.Dense(A1.RowCount, A1.ColumnCount);

        A2 = Matrix<float>.Build.Dense(outputSize, dataSize);
        A2Total = Matrix<float>.Build.Dense(A2.RowCount, A2.ColumnCount);

        W1 = Matrix<float>.Build.Dense(inputSize, hiddenSize);
        W2 = Matrix<float>.Build.Dense(hiddenSize, outputSize);

        dW1 = Matrix<float>.Build.Dense(W1.RowCount, W1.ColumnCount);
        dW2 = Matrix<float>.Build.Dense(W2.RowCount, W2.ColumnCount);

        dZ1 = Matrix<float>.Build.Dense(A1Total.RowCount, A1Total.ColumnCount);
        dZ2 = Matrix<float>.Build.Dense(A2Total.RowCount, A2Total.ColumnCount);

        for (int row = 0; row < inputSize; row++)
        {
            for (int col = 0; col < hiddenSize; col++)
            {
                W1[row, col] = UnityEngine.Random.Range(-0.5f, 0.5f);
            }
        }

        for (int row = 0; row < hiddenSize; row++)
        {
            for (int col = 0; col < outputSize; col++)
            {
                W2[row, col] = UnityEngine.Random.Range(-0.5f, 0.5f);
            }
        }

        for (int i = 0; i < hiddenSize; i++)
        {
            B1[i] = UnityEngine.Random.Range(-0.5f, 0.5f);
        }

        for (int i = 0; i < outputSize; i++)
        {
            B2[i] = UnityEngine.Random.Range(-0.5f, 0.5f);
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
            x[i] = UnityEngine.Random.Range(-10, 10);
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

        Debug.Log("input size: " + vecX.Count);
        Debug.Log("bias size: " + B1.Count);
        Debug.Log("A1Total: " + A1Total.Row(0).Count);

        Debug.Log("Weight dimensions: " + W1.Row(0).Count + " x " + W1.Column(0).Count);

        // A1 hidden x data (128 x 1) ie. 128 rows w/ 1 value :: 128 columns w/ 1 value
        // W1 input x hidden (784 x 128) ie. 784 rows w/ 128 values :: 128 columns w/ 784 values

        for (int i = 0; i < dataSize; i++)
        {
            for (int  j = 0; j < hiddenSize; j++)
            {
                A1Total[j, i] = W1.Column(j).DotProduct(vecX) + B1[j];
            }
        }
        A1 = Sigmoid(A1Total);

        for (int i = 0; i < dataSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                A2Total[j, i] = W2.Column(j).DotProduct(A1.Row(j)) + B2[j];
            }
        }
        A2 = Softmax(A2Total);

        //for (int i = 0; i < dataSize; i++)
        //{
        //    A1Total.SetColumn(i, W1.Column(i).DotProduct(vecX) + B1);
        //    A1 = Sigmoid(A1Total);
        //}

        //for (int i = 0; i < dataSize; i++)
        //{
        //    A2Total.SetRow(i, W2.Column(i).DotProduct(A1.Column(i)) + B2);
        //}
        //A2 = Softmax(A2Total);
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

    private Matrix<float> Softmax(Matrix<float> A)
    {
        //Vector<float> predictions = Vector<float>.Build.Dense(outputSize); ;
        //float sum = 0;

        //for (int i = 0; i < A.Count; i++)
        //{
        //    sum += Mathf.Exp(A[i]);
        //}

        //for (int i = 0; i < A.Count; i++)
        //{
        //    predictions[i] = Mathf.Exp(A[i]) / sum;
        //}

        //return predictions;

        float sum = 0;
        Matrix<float> softmax = A;

        for (int r = 0; r < A.RowCount; r++)
        {
            for (int c = 0; c <  A.ColumnCount; c++)
            {
                sum += Mathf.Exp(A[r, c]);
            }
        }

        for (int r = 0; r < A.RowCount; r++)
        {
            for (int c = 0; c < A.ColumnCount; c++)
            {
                softmax[r, c] = Mathf.Exp(softmax[r, c]);
            }
        }

        softmax.Divide(sum);

        return softmax;
    }

    private Matrix<float> Sigmoid(Matrix<float> x)
    {
        for (int c = 0; c < x.ColumnCount; c++)
        {
            for (int r = 0; r < x.RowCount; r++)
            {
                x[r, c] = 1 / (1 + Mathf.Exp(-x[r, c]));
            }
        }
        return x;
    }
}
