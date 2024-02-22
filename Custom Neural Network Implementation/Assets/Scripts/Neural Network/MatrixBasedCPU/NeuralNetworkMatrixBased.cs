using System.Linq;
using UnityEngine;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using Unity.VisualScripting;
using System;
using System.Collections.Generic;
using System.Collections;
using System.Threading;
using System.Threading.Tasks;
using System.Reflection;
using System.Diagnostics;

public class NeuralNetworkMatrixBased : MonoBehaviour
{
    [Header("Neural Net Hyper Params")]
    public int inputSize;
    public int hiddenSize;
    public int outputSize;

    public float learningRate;
    public int iterations;

    public bool started = false;
    public bool initialized = false;
    public bool complete = false;

    [Header("Input data")]
    private LoadImage dataSet;

    [Header("Neural Net Data")]
    private Matrix<float> W1;
    private Matrix<float> W2;

    private Vector<float> B1;
    private Vector<float> B2;

    private Thread trainingThread;

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
            trainingThread = new Thread(TrainNetwork);
            InitializeNetwork();
            UnityEngine.Debug.Log("TRAINING STARTED");
            trainingThread.Start();
        }

        if (complete)
        {
            trainingThread.Join();
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
        A1 = Matrix<float>.Build.Dense(hiddenSize, dataSet.dataNum);
        A1Total = Matrix<float>.Build.Dense(A1.RowCount, A1.ColumnCount);

        A2 = Matrix<float>.Build.Dense(outputSize, dataSet.dataNum);
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

        UnityEngine.Debug.Log("W1 Connections: " + (W1.ColumnCount * W1.RowCount));
        UnityEngine.Debug.Log("W2 Connection: " + (W2.ColumnCount * W1.RowCount));

        UnityEngine.Debug.Log("B1 Size: " + B1.Count);
        UnityEngine.Debug.Log("B2 Size: " + B2.Count);

        UnityEngine.Debug.Log("Total Connections: " + ((W1.RowCount * W1.ColumnCount) + (W2.RowCount * W2.ColumnCount) + B1.Count + B2.Count));
        UnityEngine.Debug.Log("INITIALIZATION COMPLETE");
        initialized = true;
    }

    private void TrainNetwork()
    {
        for (int i = 0; i < iterations && !complete; i++)
        {
            UnityEngine.Debug.Log("Iteration: " + i);

            var watch = Stopwatch.StartNew();

            ForwardProp();
            UnityEngine.Debug.Log("Forward Prop Complete: " + (watch.ElapsedMilliseconds) + "ms");

            watch.Restart();

            BackwardProp();
            UnityEngine.Debug.Log("Backward Prop Complete: " + (watch.ElapsedMilliseconds) + "ms");

            UpdateNetwork();

            UnityEngine.Debug.Log("Accuracy: " + Accuracy(Predictions()));

            watch.Stop();
        }
        complete = true;
    }

    private float Accuracy(Vector<float> predictions)
    {
        float correct = 0;

        for (int i = 0; i < predictions.Count; i++)
        {
            if (predictions[i].AlmostEqual(dataSet.labels[i], 0.1))
            {
                correct++;
            }
        }
        return correct / dataSet.dataNum;
    }

    private Vector<float> Predictions()
    {
        float[] predictions = new float[dataSet.dataNum];

        for (int i = 0; i < dataSet.dataNum; i++)
        {
            predictions[i] = A2.Column(i).MaximumIndex();
        }

        return Vector<float>.Build.Dense(predictions);
    }

    private void ForwardProp()
    {
        Parallel.For(0, dataSet.dataNum, i =>
        {
            A1Total.SetColumn(i, W1.LeftMultiply(dataSet.images.Column(i)) + B1);
        });
        A1 = ReLU(A1Total);

        Parallel.For(0, dataSet.dataNum, i =>
        {
            A2Total.SetColumn(i, W2.LeftMultiply(A1.Column(i)) + B2);
        });
        A2 = Softmax(A2Total);
    }

    private void BackwardProp()
    {
        dZ2 = A2 - dataSet.Y;

        Parallel.For(0, dataSet.dataNum, i =>
        {
            for (int j = 0; j < hiddenSize; j++)
            {
                dZ1[j, i] = W2.Row(j).DotProduct(dZ2.Column(i)) * ReLUDerivative(A1Total[j, i]);
            }
        });

        dB1 = (1.0f / (float)dataSet.dataNum) * dZ1.RowSums();
        dB2 = (1.0f / (float)dataSet.dataNum) * dZ2.RowSums();

        Parallel.For(0, hiddenSize, i =>
        {
            for (int j = 0; j < inputSize; j++)
            {
                dW1[j, i] = (1.0f / (float)dataSet.dataNum) * dZ1.Row(i).DotProduct(dataSet.images.Row(j));
                //UnityEngine.Debug.Log("Dw1: " + dW1[j, i] + " :: Dot Product: " + dZ1.Row(i).DotProduct(dataSet.images.Row(j)) + " :: 1 / dataset: " + (1.0f / (float)dataSet.dataNum) + " :: Final: " + (1.0f / (float)dataSet.dataNum) * dZ1.Row(i).DotProduct(dataSet.images.Row(j)));
            }
        });


        Parallel.For(0, outputSize, i =>
        {
            for (int j = 0; j < hiddenSize; j++)
            {
                dW2[j, i] = (1.0f / (float)dataSet.dataNum) * dZ2.Row(i).DotProduct(A1.Row(j));
            }
        });
    }

    private void UpdateNetwork()
    {
        //UnityEngine.Debug.Log("Prior weight: " + W1[0, 0]);
        //UnityEngine.Debug.Log("Deriv: " + dW1[0, 0]);
        //UnityEngine.Debug.Log("Predicted after: " + (W1[0,0] - (learningRate * dW1[0, 0])));

        W1 = W1 - learningRate * dW1;
        W2 = W2 - learningRate * dW2;

        //UnityEngine.Debug.Log("After: " + W1[0, 0]);

        B1 -= dB1.Multiply(learningRate);
        B2 -= dB2.Multiply(learningRate);
    }

    private Matrix<float> Softmax(Matrix<float> A)
    {
        Matrix<float> softmax = A;

        for (int i = 0; i < dataSet.dataNum; i++)
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

    private float Sigmoid(float x)
    {
        return 1 / (1 + Mathf.Exp(-x));
    }

    private Matrix<float> SigmoidDerivative(Matrix<float> ATotal)
    {
        return Sigmoid(ATotal) * (1 - Sigmoid(ATotal));
    }

    private float SigmoidDerivative(float ATotal)
    {
        return Sigmoid(ATotal) * (1 - Sigmoid(ATotal));
    }

    private Matrix<float> ReLU(Matrix<float> ATotal) 
    {
        Matrix<float> result = Matrix<float>.Build.Dense(ATotal.RowCount, ATotal.ColumnCount);

        for (int c = 0; c <  ATotal.ColumnCount; c++)
        {
            for (int r = 0; r < ATotal.RowCount; r++)
            {
                result[r, c] = ATotal[r, c] > 1.0f ? 1.0f : ATotal[r, c] < 0 ? 0.0f : ATotal[r, c];
            }
        }

        return result;
    }

    private float ReLU(float ATotal)
    {
        return ATotal > 1 ? 1.0f : ATotal < 0 ? 0.0f : ATotal;
    }

    private float ReLUDerivative(float ATotal)
    {
        return ATotal > 0 ? 1 : 0;
    }

    private void OnDestroy()
    {
        complete = true;
        trainingThread.Interrupt();
        trainingThread.Join();
    }
}