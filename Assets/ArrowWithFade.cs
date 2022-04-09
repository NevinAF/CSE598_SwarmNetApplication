using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ArrowWithFade : MonoBehaviour
{
	public float lifetime;
	public AnimationCurve normalizedLifefime;
	public float scaleMult = 1;

	private float age;
	
	void Awake()
	{
		transform.localScale = Mathf.Clamp01(normalizedLifefime.Evaluate(0)) * Vector3.one * scaleMult;
		age = 0;
	}

	// Update is called once per frame
	void Update()
	{
		age += Time.deltaTime;

		if (age > lifetime)
		{
			Destroy(gameObject);
			return;
		}
		transform.localScale = Mathf.Clamp01(normalizedLifefime.Evaluate(age/lifetime)) * Vector3.one * scaleMult;
	}
}
