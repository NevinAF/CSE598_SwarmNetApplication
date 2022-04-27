using System;
using System.IO;
using UnityEditor;
using UnityEditor.Scripting.Python;
using UnityEngine;

[CustomEditor(typeof(SelectStringWindow), true)]
public class SelectStringWindowDrawer : Editor
{
	public override void OnInspectorGUI()
	{
		SelectStringWindow stringWindow = target as SelectStringWindow;

		EditorGUILayout.LabelField("The first element on each list is used");
		SerializedProperty it = serializedObject.GetIterator();
		it.NextVisible(true);
		while (it.NextVisible(false))
			EditorGUILayout.PropertyField(it, true);


		if (GUILayout.Button("Plot Predicitons"))
		{
			stringWindow.Run(stringWindow.data_options[0], stringWindow.model_options[0]);
			stringWindow.Save();
		}

		serializedObject.ApplyModifiedProperties();
	}
}

public class SelectStringWindow : EditorWindow
{
	private ScriptableList data;
	private ScriptableList models;

	public void Save()
	{
		data.items = new System.Collections.Generic.List<string>(data_options);
		models.items = new System.Collections.Generic.List<string>(model_options);
	}

	public SelectStringWindow Init(string datapath, string modelpath)
	{
		Debug.Log(datapath + "\n" + modelpath);
		data = AssetDatabase.LoadAssetAtPath<ScriptableList>(datapath);
		data_options = data.items.ToArray();
		models = AssetDatabase.LoadAssetAtPath<ScriptableList>(modelpath);
		model_options = models.items.ToArray();

		return this;
	}

	public void Run(string data, string model)
	{
		string demoFileData = Application.dataPath + data;

		float[][][] preditions = SwarmNetAPI.Control(
			last_steps: SwarmNetAPI.ParseJsonData(
				File.ReadAllText(demoFileData), FilePlotData.FileFormat.NodeTimeSate),
			model_name: model,
			numPredictedSteps: steps
		);

		CreatePlot.PlotMatrix(preditions, $"Plot (m: {model}, d: {data})", fromVel: plotVelocity);
	}

	public string[] data_options;
	public string[] model_options;
	public bool plotVelocity;
	public int steps = 10;

	Editor editor;

	void OnGUI()
	{
		if (!editor) { editor = Editor.CreateEditor(this); }
		if (editor) { editor.OnInspectorGUI(); }
	}
}



public static class SwarmNetAPI
{
	public const string DATA_OBJECT = @"Assets/Data/TestData.asset";
	public const string MODEL_OBJECT = @"Assets/Data/TestModels.asset";

	[MenuItem("SwarmNetAPI/PlotfromModel")]
	public static void Run()
	{
		SelectStringWindow wind = EditorWindow.GetWindow<SelectStringWindow>(true, "Plot From Model");
		wind.Init(DATA_OBJECT, MODEL_OBJECT);
		wind.Show();
	}

	public static float[][][] ParseJsonData(string json, FilePlotData.FileFormat fileFormat, bool inQuotes = false, int condensedIndex = 0)
	{
		try
		{
			float[][][] data;

			if (inQuotes) json = json.Substring(1, json.Length - 2);

			if (fileFormat == FilePlotData.FileFormat.NodeTimeSate)
				data = Newtonsoft.Json.JsonConvert.DeserializeObject<float[][][]>(json);
			else
			{
				float[][][][] temp;
				temp = Newtonsoft.Json.JsonConvert.DeserializeObject<float[][][][]>(json);

				data = new float[temp.Length][][];

				if (fileFormat == FilePlotData.FileFormat.NodeCondTimeState)
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
						data[i] = new float[temp[i][condensedIndex].Length][];
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

			return data;
		}
		catch (Exception e)
		{
			Debug.LogError(condensedIndex + " " + e.Message + "\n" + e.StackTrace);
			return null;
		}
	}

	public static float[][][] Control(float[][][] last_steps, string model_name, int numPredictedSteps)
	{
		var psi = new System.Diagnostics.ProcessStartInfo();
		// point to python virtual env
		psi.FileName = @"python";

		// Provide arguments
		string jsonpath = Application.dataPath + @"\..\Dump.json";
		string script = Application.dataPath + @"\..\SwarmNet - PyTorch\control_clone_swarm_exe.py";
		string model = Application.dataPath + @"\..\SwarmNet - PyTorch\models\" + model_name;

		File.WriteAllText(jsonpath, Newtonsoft.Json.JsonConvert.SerializeObject(last_steps));


		psi.Arguments = $"\"{script}\" \"{jsonpath}\" \"{model}\" {numPredictedSteps}";

		// Process configuration
		psi.UseShellExecute = false;
		psi.CreateNoWindow = true;
		psi.RedirectStandardOutput = true;
		psi.RedirectStandardError = true;

		//Debug.Log($"Staring Cmd: {psi.FileName} {psi.Arguments}");
		using (var process = System.Diagnostics.Process.Start(psi))
		{
			string cerr = process.StandardError.ReadToEnd();
			string cout = process.StandardOutput.ReadToEnd();
			if (!string.IsNullOrEmpty(cerr))
				Debug.LogError(cerr);
			if (!string.IsNullOrEmpty(cout))
				Debug.Log(cout);

			if (process.ExitCode != 0) return null;
		}

		return ParseJsonData(File.ReadAllText(jsonpath), FilePlotData.FileFormat.NodePredictSteps, false, condensedIndex: 0);
	}
}
