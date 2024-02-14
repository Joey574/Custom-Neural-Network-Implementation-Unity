using System;
using System.Collections.Generic;

public class Node
{
    public List<Node> inputs;
    public List<float> weight;

    public float bias;
    public float output; 

    public void ComputeOutput()
    {
        for (int i = 0; i < inputs.Count; i++)
        {
            output = inputs[i].output * weight[i];
        }

        output += bias;

        output = output > 1 ? 1 : Math.Max(0, output);
    }
}
