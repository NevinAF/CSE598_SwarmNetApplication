using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class FilePlotData : ScriptableObject
{
	public enum FileFormat { NodeTimeSate = 0, NodeCondTimeState = 1, NodePredictSteps = 2 }

	public FileFormat fileFormat;
	public bool inQuotes;
	public string filename;
	public GameObject lineprefab;
}

