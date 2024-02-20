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

public class LoadImage : MonoBehaviour
{
    private const string TrainingImagePath = "Assets/Resources/Training Data/train-images.idx3-ubyte";
    private const string TrainingLabelPath = "Assets/Resources/Training Data/train-labels.idx1-ubyte";

    private const string TestingImagePath = "Assets/Resources/Testing Data/t10k-images.idx3-ubyte";
    private const string TestingLabelPath = "Assets/Resources/Testing Data/t10k-labels.idx1-ubyte";

    public Matrix<float> images;
    public Matrix<float> Y;
    public List<int> labels;
    public int imageNum;

    public bool ImagesLoaded = false;

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

        int magicNum = ReadBigInt32(reader);
        imageNum = ReadBigInt32(reader);
        int width = ReadBigInt32(reader);
        int height = ReadBigInt32(reader);

        int magicLabel = ReadBigInt32(labelReader);
        int numLabels = ReadBigInt32(labelReader);

        images = Matrix<float>.Build.Dense(width * height, imageNum);
        labels = new List<int>();

        Debug.Log("MagicNum: " + magicNum);
        Debug.Log("numOfImages: " + imageNum);
        Debug.Log("numOfLabels: " + numLabels);
        Debug.Log("width: " + width);
        Debug.Log("height: " + height);

        for (int i = 0; i < imageNum; i++)
        {
            byte[] bytes = reader.ReadBytes(width * height);
            float[] floats = new float[bytes.Length];
            Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
            images.SetColumn(i, floats);
            labels.Add(labelReader.ReadByte());
        }

        images.Divide(255);

        Debug.Log("Images Loaded");

        Debug.Log("Total Images: " + images.ColumnCount);
        Debug.Log("Total Labels: " + labels.Count);
        Debug.Log("Total pixels x image: " + images.RowCount);

        reader.Close();
        labelReader.Close();

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
