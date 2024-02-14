using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NeuralNetwork : MonoBehaviour
{
    [Header("Neural Network")]
    private OutputLayer outputLayer = new OutputLayer();
    private InputLayer inputLayer = new InputLayer();
    private HiddenLayer hiddenLayer = new HiddenLayer();

    [Header("Network Size")]
    public int inputLayerSize;
    public int hiddenLayerSize;
    public int outputLayerSize;

    void Awake()
    {
        inputLayer.InitializeLayer(inputLayerSize);
        hiddenLayer.InitializeLayer(hiddenLayerSize, inputLayerSize);
        outputLayer.InitializeLayer(outputLayerSize, hiddenLayerSize);
    }

    private void Iterate()
    {
        // TODO
        inputLayer.SetInput();

        hiddenLayer.UpdateLayer(inputLayer.GetActivation());

        outputLayer.UpdateLayer(hiddenLayer.GetActivation());
    }
}
