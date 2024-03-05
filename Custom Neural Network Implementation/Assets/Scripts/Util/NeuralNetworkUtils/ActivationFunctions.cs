using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;

public static class ActivationFunctions
{
    public static Matrix<float> ReLU(Matrix<float> Z)
    {
        return Z.Map(x => x < 0.0f ? 0.0f : x);
    }

    public static Vector<float> ReLU(Vector<float> Z)
    {
        return Z.Map(x => x < 0.0f ? 0.0f : x);
    }

    public static float ReLU(float Z)
    {
        return Z < 0.0f ? 0.0f : Z;
    }

    public static Matrix<float> ReLUDerivative(Matrix<float> Z)
    {
        return Z.Map(x => x > 0.0f ? 1.0f : 0.0f);
    }

    public static Vector<float> ReLUDerivative(Vector<float> Z)
    {
        return Z.Map(x => x > 0.0f ? 1.0f : 0.0f);
    }

    public static float ReLUDerivative(float Z)
    {
        return Z > 0.0f ? 1.0f : 0.0f;
    }

    public static Matrix<float> Sigmoid(Matrix<float> Z)
    {
        return null;
    }

}
