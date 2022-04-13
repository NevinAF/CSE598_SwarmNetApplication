using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

public class CreatePlot : EditorWindow
{
	public enum FileFormat { NodeTimeSate = 0, NodeCondTimeState = 1 }

	[MenuItem("Window/CreatePlot")]
	static void Init()
	{
		GetWindow<CreatePlot>(false, "URP->HDRP Wizard").Show();
	}

	private FileFormat fileFormat;
	private int sort_type;
	private string filepath = null;
	private string filename = "NONE.json";
	private GameObject lineprefab = null;
	private int Maxtimesteps = int.MaxValue;
	private int MaxAgents = int.MaxValue;

	void OnGUI()
	{
		GUILayout.Label("Base Settings", EditorStyles.boldLabel);
		fileFormat = (FileFormat)GUILayout.Toolbar((int)fileFormat, new string[] { "Ground Truth", "Prediction" });
		sort_type = GUILayout.Toolbar(sort_type, new string[] { "Agents", "Timestep" });

		Maxtimesteps = EditorGUILayout.IntField("Max timestep: ", Maxtimesteps);
		MaxAgents = EditorGUILayout.IntField("Max Agents: ", MaxAgents);

		filepath = EditorGUILayout.TextField("File Path", filepath);
		filename = EditorGUILayout.TextField("File Name", filename);
		lineprefab = EditorGUILayout.ObjectField("Line Prefab: ", lineprefab, typeof(GameObject), false) as GameObject;

		if (GUILayout.Button("Autoset Path")) filepath = Application.dataPath;

		GUILayout.Space(20);
		if (GUILayout.Button("Plot!"))
		{
			PlotFrom(filepath + "/" + filename, fileFormat, sort_type);
        }
    }

	private void PlotFrom(string file, FileFormat format, int sort)
    {
		float[][][] data = null;
		try
		{
			if (format == FileFormat.NodeTimeSate)
				data = Newtonsoft.Json.JsonConvert.DeserializeObject<float[][][]>(File.ReadAllText(file));
			else
			{
				Debug.Log("Reading as 4");
				float[][][][] temp = Newtonsoft.Json.JsonConvert.DeserializeObject<float[][][][]>(File.ReadAllText(file));
				data = new float[temp.Length][][];
				for (int i = 0; i < temp.Length; i++)
				{
					data[i] = new float[temp[i].Length][];
					for (int j = 0; j < temp[i].Length; j++)
					{
						data[i][j] = new float[temp[i][j][0].Length];
						for (int k = 0; k < temp[i][j][0].Length; k++)
						{
							data[i][j][k] = temp[i][j][0][k];
						}
					}
				}
			}

			if (data == null) throw new Exception();
		}
		catch (Exception e)
		{
			Debug.Log(file);
			Debug.LogError(e.Message + "\n" + e.StackTrace);
			return;
		}

		GameObject plot_go = new GameObject("Plot: " + filename);
		plot_go.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);

		if (sort == 0)
		{
			for (int i = 0; i < data.Length && i < MaxAgents; i++)
			{
				new GameObject("Agent " + i).transform.parent = plot_go.transform;
			}
		}
		else
		{
			for (int j = 0; j < data[0].Length && j < Maxtimesteps; j++)
			{
				new GameObject("Timestep " + j).transform.parent = plot_go.transform;
			}
		}
		

		Transform iparent = null;
		for (int i = 0; i < data.Length && i < MaxAgents; i++)
		{
			if (sort == 0) iparent = plot_go.transform.GetChild(i);

			for (int j = 0; j < data[i].Length && j < Maxtimesteps; j++)
			{
				if (sort == 1) iparent = plot_go.transform.GetChild(j);

				Vector3 pos = new Vector3(
					data[i][j][0],
					data[i][j][1],
					data[i][j][2]
				);
				Vector3 dir = Feesh.PVector.GetVector3(
					data[i][j][3],
					data[i][j][4],
					data[i][j][5]
				);

				GameObject go = PrefabUtility.InstantiatePrefab(lineprefab, iparent) as GameObject;
				go.name = "A" + i + " T" + j;
				go.transform.SetPositionAndRotation(pos, Quaternion.LookRotation(dir, Vector3.up));
			}
        }
    }

    public static class JsonArrayReader
    {
		public static T[] FromJsonArray<T>(string json)
        {
			string newJson = "{ \"array\": " + json + "}";
			Wrapper<T> wrapper = JsonUtility.FromJson<Wrapper<T>>(newJson);
			return wrapper.array;
        }

		[Serializable]
		public class Wrapper<T> { public T[] array; }
	}
}
