using UnityEngine;

public class IA_Vie : MonoBehaviour
{

    [Header("IA Script")]
     private IA ia_Main_Script;

    [Header("IA Health")]
    //SO can modify letter if no one shot
     private int max_Health = 100;
     private int health;




    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
    }

    // Update is called once per frame
    public void Take_Damage(int d)
    {
        health = health - (d);
        if(health <= 0)
        {
            ia_Main_Script.Die();
        }
    }
    public void Set_Up(IA ia_Main_Script_,int max_Healf_)
    {
        ia_Main_Script = ia_Main_Script_;
        health = max_Healf_;
    }
}
