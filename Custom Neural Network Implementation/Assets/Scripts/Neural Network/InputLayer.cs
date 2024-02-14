using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class InputLayer
{
    public int size;
    public List<Node> nodes = new List<Node>();

    public void InitializeLayer(int size)
    {
        this.size = size;

        for (int i = 0; i < size; i++)
        {
            nodes.Add(new Node(null, Random.Range(0.0f, 1.0f)));
        }
    }

    public void SetInput()
    {
        // TODO
        for (int i = 0; i < nodes.Count; i++)
        {
            nodes[i].SetActivation(0);
        }
    }

}
