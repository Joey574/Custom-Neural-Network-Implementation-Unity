using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Node
{
    public List<float> weights = new List<float>();
    public float bias;

    public float activation;

    public Node(List<float> weights, float bias)
    {
        this.weights = weights;
        this.bias = bias;
    }

    public void UpdateActivation(List<float> input)
    {
        activation = 0;

        for (int i = 0; i < input.Count; i++) 
        {
            activation = input[i] * weights[i];
        }

        activation += bias;
        activation = SigmoidFunction(activation);
    }

    public void SetActivation(float activation)
    {
        this.activation = activation;
    }

    public float SigmoidFunction(float input)
    {
        return 1 / (1 - Mathf.Exp(-activation));
    }

}
