using UnityEngine;

public class IA_Utils : MonoBehaviour
{
    public enum Type
    {
        Helmet,
        Vest,
        Visor,
        Weapon,
    }
    public enum Equipement_Type
    {
        Nothing,
        Light,
        Medium,
        Heavy,
        Random
    }
    [System.Serializable]
    public class EquipmentSet
    {
        public GameObject[] Light;
        public GameObject[] Medium;
        public GameObject[] Heavy;
    }
    [System.Serializable]
    public class WeaponSet
    {
        public Gun_Scriptable[] Light;
        public Gun_Scriptable[] Medium;
        public Gun_Scriptable[] Heavy;
    }


}
