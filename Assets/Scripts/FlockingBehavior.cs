using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public abstract class FlockingBehavior : ScriptableObject
{
    public virtual bool UsesNearby() { return true; }
    public abstract Vector3 CalculateDirection (Feesh feesh,  List<Transform> nearby, Flock flock); 
}
