using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

public class CreatePlot : EditorWindow
{
	public enum FileFormat { NodeTimeSate }

	[MenuItem("Window/CreatePlot")]
	static void Init()
	{
		GetWindow<CreatePlot>(false, "URP->HDRP Wizard").Show();
	}

	private string filepath = null;
	private string filename = "NONE.json";
	private GameObject lineprefab = null;

	void OnGUI()
	{
		GUILayout.Label("Base Settings", EditorStyles.boldLabel);

		filepath = EditorGUILayout.TextField("File Path", filepath);
		filename = EditorGUILayout.TextField("File Name", filename);
		lineprefab = EditorGUILayout.ObjectField("Line Prefab: ", lineprefab, typeof(GameObject), false) as GameObject;

		if (GUILayout.Button("Autoset Path")) filepath = Application.dataPath;

		GUILayout.Space(20);
		if (GUILayout.Button("Plot!"))
		{
			PlotFrom(filepath + "/" + filename, FileFormat.NodeTimeSate);
        }
    }

	private void PlotFrom(string file, FileFormat format)
    {
		float[][][] data = null;
		try
		{
            data = Newtonsoft.Json.JsonConvert.DeserializeObject<float[][][]>(File.ReadAllText(file));

			if (data == null) throw new Exception();
		}
		catch (Exception e)
		{
			Debug.Log(file);
			Debug.LogError(e.Message + "\n" + e.StackTrace);
			return;
		}

		GameObject plot_go = new GameObject("Plot");
		plot_go.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);

		for (int i = 0; i < data.Length; i++)
		{
			for (int j = 0; j < data[i].Length; j++)
			{
				Debug.Log(i + "," + j);
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
				Instantiate(lineprefab, pos, Quaternion.LookRotation(dir, Vector3.up), plot_go.transform);
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
