using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NeuralNetwork : MonoBehaviour
{
    [Header("Neural Network")]
    private List<OutputNode> OutputLayer = new List<OutputNode>();
    private List<InputNode> InputLayer = new List<InputNode>();
    private List<HiddenNode> HiddenLayer = new List<HiddenNode>();

    [Header("Network Size")]
    public int inputLayerSize;
    public int hiddenLayerSize;
    public int outputLayerSize;

    void Awake()
    {
        for (int i = 0; i < inputLayerSize; i++) 
        {
            InputLayer.Add(new InputNode());
        }

        for (int i = 0; i < hiddenLayerSize; i++)
        {
            HiddenLayer.Add(new HiddenNode());
        }

        for (int i = 0; i < outputLayerSize; i++)
        {
            OutputLayer.Add(new OutputNode());
        }
    }

    private void Iterate()
    {
    
    }
}
