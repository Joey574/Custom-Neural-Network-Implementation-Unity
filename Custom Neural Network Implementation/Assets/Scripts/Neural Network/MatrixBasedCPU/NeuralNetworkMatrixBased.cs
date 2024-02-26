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
using UnityEngine.UIElements;
using Unity.VisualScripting.Antlr3.Runtime;
using Unity.Mathematics;

public class NeuralNetworkMatrixBased : MonoBehaviour
{
    [Header("Neural Net Hyper Params")]
    public int inputSize;
    public int outputSize;
    public List<int> hiddenSize;

    public float learningRate;
    public int iterations;

    [Header("Status")]
    public bool started = false;
    public bool initialized = false;
    public bool complete = false;
    public bool testingComplete = false;

    [Header("Save Data")]
    public bool Save;
    public string SaveName;
    public string LoadName;

    [Header("Input data")]
    private LoadImage dataSet;

    [Header("Neural Net Data")]
    private List<Matrix<float>> weights;
    private List<Vector<float>> biases;

    private Thread trainingThread;
    private Thread testingThread;

    [Header("Neural Net Outputs")]
    private List<Matrix<float>> A;
    private List<Matrix<float>> ATotal;

    [Header("Derivative Values")]
    private List<Matrix<float>> dWeights;
    private List<Matrix<float>> dTotal;

    private List<Vector<float>> dBiases;

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
            complete = false;
            trainingThread.Join();
            testingThread = new Thread(TestNetwork);
            testingThread.Start();

            if (Save)
            {
                SaveNetwork.SaveNeuralNetwork(weights, biases, SaveName);
            }
        }

        if (testingComplete)
        {
            testingThread.Join();
        }
    }

    private void InitializeNetwork()
    {
        biases = new List<Vector<float>>();
        dBiases = new List<Vector<float>>();
        weights = new List<Matrix<float>>();
        dWeights = new List<Matrix<float>>();

        // Biases
        for (int i = 0; i < hiddenSize.Count; i++)
        {
            biases.Add(Vector<float>.Build.Dense(Array.ConvertAll(new float[hiddenSize[i]], _ => UnityEngine.Random.Range(-0.5f, 0.5f))));
        }
        biases.Add(Vector<float>.Build.Dense(Array.ConvertAll(new float[outputSize], _ => UnityEngine.Random.Range(-0.5f, 0.5f))));

        // Weights
        weights.Add(Matrix<float>.Build.Dense(inputSize, hiddenSize[0], Array.ConvertAll(new float[inputSize * hiddenSize[0]], _ => UnityEngine.Random.Range(-0.5f, 0.5f))));
        for (int i = 0; i < hiddenSize.Count - 1; i++)
        {
            weights.Add(Matrix<float>.Build.Dense(hiddenSize[i], hiddenSize[i + 1], Array.ConvertAll(new float[hiddenSize[i] * hiddenSize[i + 1]], _ => UnityEngine.Random.Range(-0.5f, 0.5f))));
        }
        weights.Add(Matrix<float>.Build.Dense(hiddenSize[hiddenSize.Count - 1], outputSize, Array.ConvertAll(new float[hiddenSize[hiddenSize.Count - 1] * outputSize], _ => UnityEngine.Random.Range(-0.5f, 0.5f))));

        // Derivatives
        for (int i = 0; i < weights.Count; i++)
        {
            dWeights.Add(Matrix<float>.Build.Dense(weights[i].RowCount, weights[i].ColumnCount));
        }

        for (int i = 0; i < biases.Count; i++)
        {
            dBiases.Add(Vector<float>.Build.Dense(biases[i].Count));
        }

        InitializeResultMatrices(dataSet.dataNum);

        // Display
        int connections = 0;

        for (int i = 0; i < weights.Count; i++)
        {
            UnityEngine.Debug.Log("W" + i + " Connections: " + (weights[i].RowCount * weights[i].ColumnCount));
            connections += weights[i].RowCount * weights[i].ColumnCount;
        }

        for (int i = 0; i < biases.Count; i++)
        {
            UnityEngine.Debug.Log("B" + i + " Size: " + biases[i].Count);
            connections += biases[i].Count;
        }

        UnityEngine.Debug.Log("Total Connections: " + connections);
        UnityEngine.Debug.Log("Predicted Size of File: " + (((sizeof(float) * connections) + connections) * 2) / 1000000.0f + "mb");
        UnityEngine.Debug.Log("INITIALIZATION COMPLETE");
        initialized = true;
    }

    private void InitializeResultMatrices(int size)
    {
        A = new List<Matrix<float>>();
        ATotal = new List<Matrix<float>>();
        dTotal = new List<Matrix<float>>();

        // Values
        for (int i = 0; i < hiddenSize.Count; i++)
        {
            A.Add(Matrix<float>.Build.Dense(hiddenSize[i], size));
            ATotal.Add(Matrix<float>.Build.Dense(A[i].RowCount, A[i].ColumnCount));
        }

        A.Add(Matrix<float>.Build.Dense(outputSize, size));
        ATotal.Add(Matrix<float>.Build.Dense(A[A.Count - 1].RowCount, A[A.Count - 1].ColumnCount));

        // Derivatives
        for (int i = 0; i < ATotal.Count; i++)
        {
            dTotal.Add(Matrix<float>.Build.Dense(ATotal[i].RowCount, ATotal[i].ColumnCount));
        }
    }

    private void TrainNetwork()
    {
        for (int i = 0; i < iterations && !complete; i++)
        {
            UnityEngine.Debug.Log("Iteration: " + i);

            var watch = Stopwatch.StartNew();

            ForwardProp(dataSet.images);
            UnityEngine.Debug.Log("Forward Prop Complete: " + (watch.ElapsedMilliseconds) + "ms");

            watch.Restart();

            BackwardProp();
            UnityEngine.Debug.Log("Backward Prop Complete: " + (watch.ElapsedMilliseconds) + "ms");

            UpdateNetwork();

            UnityEngine.Debug.Log("Accuracy: " + Accuracy(Predictions(dataSet.dataNum), dataSet.labels));

            watch.Stop();
        }
        complete = true;
    }

    private void TestNetwork()
    {
        InitializeResultMatrices(dataSet.TestingImages.ColumnCount);

        ForwardProp(dataSet.TestingImages);
        UnityEngine.Debug.Log("Final Accuracy: " + Accuracy(Predictions(dataSet.TestingImages.ColumnCount), dataSet.TestingLabels));
        testingComplete = true;
    }

    private float Accuracy(Vector<float> predictions, List<int> labels)
    {
        float correct = 0;

        for (int i = 0; i < predictions.Count; i++)
        {
            if (predictions[i].AlmostEqual(labels[i], 0.1))
            {
                correct++;
            }
        }
        return correct / predictions.Count;
    }

    private Vector<float> Predictions(int len)
    {
        float[] predictions = new float[len];

        for (int i = 0; i < len; i++)
        {
            predictions[i] = A[A.Count - 1].Column(i).MaximumIndex();
        }

        return Vector<float>.Build.Dense(predictions);
    }

    private void ForwardProp(Matrix<float> input)
    {
        Parallel.For(0, input.ColumnCount, i =>
        {
            ATotal[0].SetColumn(i, weights[0].LeftMultiply(input.Column(i)) + biases[0]);
        });
        A[0] =ReLU(ATotal[0]);

        for (int x = 1; x < A.Count; x++)
        {
            Parallel.For(0, input.ColumnCount, i =>
            {
                ATotal[x].SetColumn(i, weights[x].LeftMultiply(A[x - 1].Column(i)) + biases[x]);
            });
            A[x] = x < A.Count - 1 ? ReLU(ATotal[x]) : Softmax(ATotal[x]);
        }
    }

    private void BackwardProp()
    {
        dTotal[dTotal.Count - 1] = A[A.Count - 1] - dataSet.Y;
        Parallel.For(0, dTotal.Count - 1, i =>
        {
            dTotal[i] = (weights[i + 1] * dTotal[dTotal.Count - 1]) * ReLUDerivative(ATotal[i]);
        });

        dWeights[0] = (1.0f / (float)dataSet.dataNum) * (dTotal[0] * dataSet.images);
        Parallel.For(1, dWeights.Count, i =>
        {
            dWeights[i] = (1.0f / (float)dataSet.dataNum) * (dTotal[i] * A[i - 1]);
        });

        Parallel.For(0, biases.Count, i =>
        {
            dBiases[i] = (1.0f / (float)dataSet.dataNum) * dTotal[i].RowSums();
        });

        //dZ2 = A2 - dataSet.Y;

        //Parallel.For(0, dataSet.dataNum, i =>
        //{
        //    for (int j = 0; j < hiddenSize; j++)
        //    {
        //        dZ1[j, i] = W2.Row(j).DotProduct(dZ2.Column(i)) * ReLUDerivative(A1Total[j, i]);
        //    }
        //});

        //Parallel.For(0, hiddenSize, i =>
        //{
        //    for (int j = 0; j < inputSize; j++)
        //    {
        //        dW1[j, i] = (1.0f / (float)dataSet.dataNum) * dZ1.Row(i).DotProduct(dataSet.images.Row(j));
        //    }
        //});

        //Parallel.For(0, outputSize, i =>
        //{
        //    for (int j = 0; j < hiddenSize; j++)
        //    {
        //        dW2[j, i] = (1.0f / (float)dataSet.dataNum) * dZ2.Row(i).DotProduct(A1.Row(j));
        //    }
        //});
    }

    private void UpdateNetwork()
    {
        Parallel.For(0, weights.Count, i =>
        {
            weights[i] = weights[i] - learningRate * dWeights[i];

        });

        Parallel.For(0, weights.Count, i =>
        {
            biases[i] -= dBiases[i].Multiply(learningRate);
        });
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

    private Matrix<float> SigmoidDerivative(Matrix<float> A)
    {
        return Sigmoid(A) * (1 - Sigmoid(A));
    }

    private float SigmoidDerivative(float A)
    {
        return Sigmoid(A) * (1 - Sigmoid(A));
    }

    private Matrix<float> ReLU(Matrix<float> A) 
    {
        Matrix<float> result = Matrix<float>.Build.Dense(A.RowCount, A.ColumnCount);

        for (int c = 0; c <  A.ColumnCount; c++)
        {
            for (int r = 0; r < A.RowCount; r++)
            {
                result[r, c] = A[r, c] > 1.0f ? 1.0f : A[r, c] < 0 ? 0.0f : A[r, c];
            }
        }

        return result;
    }

    private float ReLU(float A)
    {
        return A > 1 ? 1.0f : A < 0 ? 0.0f : A;
    }

    private Matrix<float> ReLUDerivative(Matrix<float> A)
    {
        Matrix<float> matrix = Matrix<float>.Build.Dense(A.RowCount, A.ColumnCount);

        for (int c = 0; c < A.ColumnCount; c++)
        {
            for (int r = 0; r < A.RowCount; r++)
            {
                matrix[r, c] = A[r, c] > 0 ? 1.0f : 0;
            }
        }

        return matrix;
    }

    private float ReLUDerivative(float A)
    {
        return A > 0 ? 1 : 0;
    }

    private void OnDestroy()
    {
        if (!complete && Save)
        {
            SaveNetwork.SaveNeuralNetwork(weights, biases, SaveName);
        }

        complete = true;
        trainingThread.Interrupt();
        trainingThread.Join();
        testingThread.Interrupt();
        testingThread.Join();
    }
}