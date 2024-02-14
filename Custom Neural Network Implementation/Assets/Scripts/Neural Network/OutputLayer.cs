using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class OutputLayer
{
    private int size;
    public List<Node> nodes = new List<Node>();

    public void InitializeLayer(int size, int lastSize)
    {
        this.size = size;

        for (int i = 0; i < size; i++)
        {
            List<float> randWeight = new List<float>();

            for (int x = 0; x < lastSize; x++)
            {
                randWeight.Add(Random.Range(0.0f, 1.0f));
            }

            nodes.Add(new Node(randWeight, Random.Range(0.0f, 1.0f)));
        }
    }

    public void UpdateLayer(List<float> input)
    {
        foreach (Node node in nodes)
        {
            node.UpdateActivation(input);
        }
    }

    public List<float> GetActivation()
    {
        List<float> activation = new List<float>();

        foreach (Node node in nodes)
        {
            activation.Add(node.activation);
        }

        return activation;
    }
}
