using UnityEngine;

[CreateAssetMenu(menuName = "ScriptableList")]
public class ScriptableList : ScriptableObject
{
	[SerializeField]
	public System.Collections.Generic.List<string> items;
}