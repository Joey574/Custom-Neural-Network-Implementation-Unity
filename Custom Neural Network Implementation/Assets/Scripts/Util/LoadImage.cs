using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class LoadImage : MonoBehaviour
{
    private const string MNISTImagesPath = "Assets/Resources/Training Data/train-images.idx3-ubyte";
    private const string MNISTLabelsPath = "Assets/Resources/Training Data/train-labels.idx1-ubyte";

    private void Start()
    {
        // Load MNIST data
        List<byte[]> images = LoadMNISTImages(MNISTImagesPath);
        List<byte> labels = LoadMNISTLabels(MNISTLabelsPath);

        // Process the data (e.g., convert to grayscale textures, normalize, etc.)
        // Your custom processing logic goes here
        // For demonstration purposes, let's print the first label and image size:
        Debug.Log($"Number of images: {images.Count}");
        Debug.Log($"First label: {labels[0]}");
        Debug.Log($"First image size: {images[0].Length} bytes");
    }

    private List<byte[]> LoadMNISTImages(string path)
    {
        List<byte[]> imageList = new List<byte[]>();
        using (BinaryReader reader = new BinaryReader(new FileStream(path, FileMode.Open)))
        {
            // Read MNIST header information
            int magicNumber = reader.ReadInt32();
            int numImages = reader.ReadInt32();
            int numRows = reader.ReadInt32();
            int numCols = reader.ReadInt32();

            Debug.Log(numImages);

            for (int i = 0; i < numImages; i++)
            {
                byte[] imageBytes = reader.ReadBytes(numRows * numCols);
                imageList.Add(imageBytes);
            }
        }
        return imageList;
    }

    private List<byte> LoadMNISTLabels(string path)
    {
        List<byte> labelList = new List<byte>();
        using (BinaryReader reader = new BinaryReader(new FileStream(path, FileMode.Open)))
        {
            // Read MNIST header information
            int magicNumber = reader.ReadInt32();
            int numLabels = reader.ReadInt32();

            for (int i = 0; i < numLabels; i++)
            {
                byte label = reader.ReadByte();
                labelList.Add(label);
            }
        }
        return labelList;
    }
}
