using System.Linq;
using UnityEngine;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using Unity.VisualScripting;
using System;
using System.Collections.Generic;

public class NeuralNetworkMatrixBased : MonoBehaviour
{
    [Header("Neural Net Hyper Params")]
    public int inputSize;
    public int hiddenSize;
    public int outputSize;

    public float learningRate;
    public int iterations;

    public bool started = false;

    [Header("Input data")]
    private LoadImage dataSet;

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
        dataSet = gameObject.GetComponent<LoadImage>();
    }

    void Update()
    {
        if (dataSet.ImagesLoaded && !started)
        {
            started = true;
            InitializeNetwork();
            StartCoroutine(nameof(TrainNetwork));
        }
    }

    private void InitializeNetwork()
    {
        // Vectors
        B1 = Vector<float>.Build.Dense(hiddenSize);
        B2 = Vector<float>.Build.Dense(outputSize);

        dB1 = Vector<float>.Build.Dense(B1.Count);
        dB2 = Vector<float>.Build.Dense(B2.Count);

        // Matrices
        A1 = Matrix<float>.Build.Dense(hiddenSize, dataSet.imageNum);
        A1Total = Matrix<float>.Build.Dense(A1.RowCount, A1.ColumnCount);

        A2 = Matrix<float>.Build.Dense(outputSize, dataSet.imageNum);
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

    private void TrainNetwork()
    {
        for (int i = 0; i < iterations; i++)
        {
            Vector<float> input = Vector<float>.Build.Dense(inputSize);

            ForwardProp(input);
            BackwardProp(input, i);
            UpdateNetwork();

            if (i % 10 == 0)
            {
                Debug.Log("Iteration: " + i);
                Debug.Log("Accuracy: " + Accuracy(Predictions()));
            }
        }
    }

    private void TestNetwork()
    {
        Vector<float> x = Vector<float>.Build.Dense(inputSize);

        for (int i = 0; i < inputSize; i++)
        {
            x[i] = UnityEngine.Random.Range(-10, 10);
        }

        ForwardProp(x);
        DisplayProbabilities();
    }

    private void DisplayProbabilities()
    {
        // A2 output x data (10 x 1) ie. 10 rows w/ 1 value :: 1 column w/ 10 values

        Debug.Log("Probabilities of last input: " + A2.Column(A2.ColumnCount - 1).ToCommaSeparatedString() + " :: Sum: " + A2.Column(A2.ColumnCount - 1).Sum());
    }

    private float Accuracy(Vector<int> predictions)
    {
        float correct = 0;

        for (int i = 0; i < predictions.Count; i++)
        {
            if (predictions[i] == dataSet.labels[i])
            {
                correct++;
            }
        }
        return correct / dataSet.imageNum;
    }

    private Vector<int> Predictions()
    {
        int[] prediction = new int[dataSet.imageNum];

        for (int i = 0; i < dataSet.imageNum; i++)
        {
            prediction[i] = A2.Row(i).MaximumIndex();
        }

        return Vector<int>.Build.Dense(prediction);
    }

    private void ForwardProp(Vector<float> x)
    {
        // A1 hidden x data (128 x 1) ie. 128 rows w/ 1 value :: 1 column w/ 128 values
        // W1 input x hidden (784 x 128) ie. 784 rows w/ 128 values :: 128 columns w/ 784 values

        for (int i = 0; i < dataSet.imageNum; i++)
        {
            for (int  j = 0; j < hiddenSize; j++)
            {
                A1Total[j, i] = W1.Column(j).DotProduct(x) + B1[j];
            }
        }
        A1 = Sigmoid(A1Total);

        for (int i = 0; i < dataSet.imageNum; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                A2Total[j, i] = W2.Column(j).DotProduct(A1.Column(i)) + B2[j];
            }
        }
        A2 = Softmax(A2Total);
    }

    private void BackwardProp(Vector<float> x, int index)
    {
        Vector<float> y = Vector<float>.Build.Dense(outputSize);
        y[dataSet.labels[index]] = 1;

        for (int i = 0; i < dataSet.imageNum; i++)
        {
            dZ2.SetColumn(i, A2.Column(i) - y);
        }

        //      dW2 = 1 / m * dZ2.dot(A1.T)
        //      db2 = 1 / m * np.sum(dZ2)
        //      dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
        //      dW1 = 1 / m * dZ1.dot(X.T)
        //      db1 = 1 / m * np.sum(dZ1)

        dW2 = 1 / dataSet.imageNum * dZ2.Multiply(A1);
        dB2 = 1 / dataSet.imageNum * dZ2.ColumnSums();
        dZ1 = W2.PointwiseMultiply(dZ2) * SigmoidDerivative(A1Total);
        dW1 = 1 / dataSet.imageNum * dZ1.Multiply(x.ToRowMatrix());
        dB1 = 1/ dataSet.imageNum * dZ1.ColumnSums();
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
        Matrix<float> softmax = A;

        for (int i = 0; i < dataSet.imageNum; i++)
        {
            float sum = 0;

            for (int r = 0; r < A.RowCount; r++)
            {
                sum += Mathf.Exp(A[r, i]);
            }

            for (int r = 0; r < A.RowCount; r++)
            {
                softmax[r, i] = Mathf.Exp(softmax[r, i]) / sum;
            }
        }
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

    private Matrix<float> SigmoidDerivative(Matrix<float> ATotal)
    {
        return Sigmoid(ATotal) * (1 - Sigmoid(ATotal));
    }
}