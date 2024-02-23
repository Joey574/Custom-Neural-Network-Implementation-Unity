using System;
using System.IO;
using UnityEngine;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using System.Linq;
using System.Collections.Generic;
using System.Collections;
using System.Threading;
using System.Data;
using UnityEngine.XR;

public class LoadImage : MonoBehaviour
{
    private const string TrainingImagePath = "Assets/Resources/Training Data/train-images.idx3-ubyte";
    private const string TrainingLabelPath = "Assets/Resources/Training Data/train-labels.idx1-ubyte";

    private const string TestingImagePath = "Assets/Resources/Testing Data/t10k-images.idx3-ubyte";
    private const string TestingLabelPath = "Assets/Resources/Testing Data/t10k-labels.idx1-ubyte";

    [Header("Training data")]
    public Matrix<float> images;
    public Matrix<float> Y;
    public List<int> labels;
    public int dataNum;
    private int imageNum;

    public bool ImagesLoaded = false;

    [Header("Testing data")]
    public Matrix<float> TestingImages;
    public List<int> TestingLabels;

    void Awake()
    {
        Thread t = new Thread(LoadImages);
        t.Start();

        IEnumerator enumerator = JoinThread(t);
        StartCoroutine(enumerator);
    }
    
    private void LoadImages()
    {
        BinaryReader reader = new BinaryReader(new FileStream(TrainingImagePath, FileMode.Open));
        BinaryReader labelReader = new BinaryReader(new FileStream(TrainingLabelPath, FileMode.Open));

        BinaryReader TestingReader = new BinaryReader(new FileStream(TestingImagePath, FileMode.Open));
        BinaryReader TestingLabelReader = new BinaryReader(new FileStream(TestingLabelPath, FileMode.Open));

        int magicNum = ReadBigInt32(reader);
        imageNum = ReadBigInt32(reader);
        int width = ReadBigInt32(reader);
        int height = ReadBigInt32(reader);

        magicNum = ReadBigInt32(TestingReader);
        int testingNum = ReadBigInt32(TestingReader);
        ReadBigInt32(TestingReader); // width x height should be the same
        ReadBigInt32(TestingReader);

        int magicLabel = ReadBigInt32(labelReader);
        int numLabels = ReadBigInt32(labelReader);

        ReadBigInt32(TestingLabelReader); // Discard testingLabel data
        ReadBigInt32(TestingLabelReader);

        images = Matrix<float>.Build.Dense(width * height, dataNum);
        TestingImages = Matrix<float>.Build.Dense(width * height, testingNum);

        labels = new List<int>();
        TestingLabels = new List<int>();

        Debug.Log("numOfImages: " + imageNum);
        Debug.Log("numOfLabels: " + numLabels);

        for (int i = 0; i < dataNum && i < imageNum; i++)
        {
            byte[] bytes = reader.ReadBytes(width * height);
            float[] floats = new float[bytes.Length];
            bytes.CopyTo(floats, 0);
            images.SetColumn(i, floats);

            labels.Add(labelReader.ReadByte());
        }

        for (int i = 0; i < testingNum; i++)
        {
            byte[] bytes = TestingReader.ReadBytes(width * height);
            float[] floats = new float[bytes.Length];
            bytes.CopyTo(floats, 0);
            TestingImages.SetColumn(i, floats);

            TestingLabels.Add(TestingLabelReader.ReadByte());
        }           

        images = images.Divide(255);
        TestingImages = TestingImages.Divide(255);

        Debug.Log("Images Loaded");

        reader.Close();
        labelReader.Close();

        TestingReader.Close();
        TestingLabelReader.Close();

        Y = Matrix<float>.Build.Dense(10, labels.Count);

        for (int i = 0; i < labels.Count; i++)
        {
            Vector<float> y = Vector<float>.Build.Dense(10);
            y[labels[i]] = 1;
            Y.SetColumn(i, y);
        }      

        ImagesLoaded = true;
    }

    private IEnumerator JoinThread(Thread t)
    {
        while (!ImagesLoaded)
        {
            yield return null;
        }
        t.Join();
    }

    private int ReadBigInt32(BinaryReader br)
    {
        var bytes = br.ReadBytes(sizeof(Int32));
        if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
        return BitConverter.ToInt32(bytes, 0);
    }
}
