using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

public class CreatePlot : EditorWindow
{
	public enum SortType { Agent, Timesteps }

	[MenuItem("Assets/Create/FilePlotData")]
	public static void CreateMyAsset()
	{
		CreateAssetWithName();
	}

	public static void CreateAssetWithName(string filename = "NewScripableObject")
	{
		string filepath = "Assets/FilePlotData/" + filename + ".asset";

		FilePlotData asset = CreateInstance<FilePlotData>();

		Directory.CreateDirectory(Path.GetDirectoryName(filepath));

		asset.filename = Path.GetFileName(filename);
		AssetDatabase.CreateAsset(asset, filepath);
		AssetDatabase.SaveAssets();

		EditorUtility.FocusProjectWindow();

		Selection.activeObject = asset;
	}

	[MenuItem("Window/CreatePlot")]
	static void Init()
	{
		GetWindow<CreatePlot>(false, "URP->HDRP Wizard").Show();
	}

	public FilePlotData[] filedatas;
	public string root_filepath;
	public SortType sort_type;
	public int condensedIndex;
	public int Maxtimesteps = int.MaxValue;
	public int MaxAgents = int.MaxValue;
	public string newObjectName;

	Editor editor;

	void OnGUI()
	{
		if (!editor) { editor = Editor.CreateEditor(this); }
		if (editor) { editor.OnInspectorGUI(); }
	}

	internal void PlotFrom(FilePlotData filedata)
	{
		string file = root_filepath + "/" + filedata.filename;
		int sort = (int)sort_type;
		float[][][] data = null;

		try
		{
			string json = File.ReadAllText(file);
			if (filedata.inQuotes) json = json.Substring(1, json.Length - 2);
			
			if (filedata.fileFormat == FilePlotData.FileFormat.NodeTimeSate)
				data = Newtonsoft.Json.JsonConvert.DeserializeObject<float[][][]>(json);
			else
			{
				float[][][][] temp;
				temp = Newtonsoft.Json.JsonConvert.DeserializeObject<float[][][][]>(json);

				data = new float[temp.Length][][];

				if (filedata.fileFormat == FilePlotData.FileFormat.NodeCondTimeState)
				{
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
				else
				{
					for (int i = 0; i < temp.Length; i++)
					{
						data[i] = new float[temp[i].Length][];
						string s = "";
						for (int j = 0; j < temp[i][condensedIndex].Length; j++)
						{
							s += temp[i][condensedIndex][j].Length + "|";
							data[i][j] = new float[temp[i][condensedIndex][j].Length];
							for (int k = 0; k < temp[i][condensedIndex][j].Length; k++)
							{
								data[i][j][k] = temp[i][condensedIndex][j][k];
							}
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

		// DEBUG THE DATA ARRAY!
		// File.WriteAllText(Application.dataPath + "/debug.json", Newtonsoft.Json.JsonConvert.SerializeObject(data, Newtonsoft.Json.Formatting.Indented));

		GameObject plot_go = new GameObject("Plot: " + filedata.filename);
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
			if (data[i] == null) continue;

			if (sort == 0) iparent = plot_go.transform.GetChild(i);

			for (int j = 0; j < data[i].Length && j < Maxtimesteps; j++)
			{
				if (data[i][j] == null) continue;

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

				GameObject go = PrefabUtility.InstantiatePrefab(filedata.lineprefab, iparent) as GameObject;
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

[CustomEditor(typeof(CreatePlot), true)]
public class CreatePlotDrawer : Editor
{
	public override void OnInspectorGUI()
	{
		CreatePlot ploter = target as CreatePlot;

		SerializedProperty it = serializedObject.GetIterator();
		it.NextVisible(true);
		while (it.NextVisible(false))
			EditorGUILayout.PropertyField(it, true);


		if (GUILayout.Button("Autoset Path")) serializedObject.FindProperty("root_filepath").stringValue = Application.dataPath;
		if (GUILayout.Button("Create New Fileasset")) CreatePlot.CreateAssetWithName(ploter.newObjectName);
		GUILayout.Space(20);
		if (GUILayout.Button("Plot First!"))
		{
			ploter.PlotFrom(ploter.filedatas[0]);
		}

		if (GUILayout.Button("Plot All!"))
		{
			for (int i = 0; i < ploter.filedatas.Length; i++)
			{
				ploter.PlotFrom(ploter.filedatas[i]);
			}
		}

		serializedObject.ApplyModifiedProperties();
	}
}
