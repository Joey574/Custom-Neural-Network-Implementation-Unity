using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;
using System.IO;

public static class SaveNetwork
{
   public static void SaveNeuralNetwork(List<Matrix<float>> weights, List<Vector<float>> biases, string name)
    {
        using (StreamWriter sw = new StreamWriter("Assets\\SavedNetworks\\" + name + ".txt"))
        {
            sw.Write("$");

            for (int i = 0; i < weights.Count; i++)
            {
                for (int c = 0; c < weights[i].ColumnCount; c++)
                {
                    for (int r = 0; r < weights[i].RowCount; r++)
                    {
                        sw.Write(weights[i][r, c]);

                        if (r < weights[i].RowCount - 1)
                        {
                            sw.Write(",");
                        }
                    }
                    sw.WriteLine();
                }

                if (i < weights.Count - 1)
                {
                    sw.Write("$");
                }
            }

            sw.Write("#");

            for (int i = 0; i < biases.Count; i++)
            {
                for (int x = 0; x < biases[i].Count; x++)
                {
                    sw.Write(biases[i][x]);

                    if (x < biases[i].Count - 1) 
                    {
                        sw.Write(",");
                    }
                }
                sw.WriteLine();

                if (i < biases.Count - 1)
                {
                    sw.Write("#");
                }
            }
        }

        Debug.Log("Network Saved");
    }
}
