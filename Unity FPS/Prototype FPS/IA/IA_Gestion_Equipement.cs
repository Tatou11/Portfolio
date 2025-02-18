using UnityEngine;

public class IA_Gestion_Equipement : IA_Utils
{
    [Header("Equipment and Weapon Management")]
    [SerializeField] private EquipmentSet Helmet;
    [SerializeField] private EquipmentSet Vest;
    [SerializeField] private EquipmentSet Visor;
    [SerializeField] private WeaponSet Weapon;


    [Header("Variables")]
    private Equipement_Type helmet_Preset;
    private Equipement_Type vest_Preset;
    private Equipement_Type visor_Preset;
    private Equipement_Type weapon_Preset;

    private GameObject helmet_Selected;
    private GameObject vest_Selected;
    private GameObject visor_Selected;
    private Gun_Scriptable weapon_Selected;

    [Header("Equipment and Weapon Placement")]

    private GameObject helmet_Empty;
    private GameObject visor_Empty;
    private GameObject vest_Empty;
    private GameObject weapon_Empty;

    // Start is called once before the first execution of Update after the MonoBehaviour is created

    private void Equipe_Now()
    {
        helmet_Selected = Equipement_Choice(helmet_Preset, Helmet);
        visor_Selected = Equipement_Choice(visor_Preset, Visor);
        vest_Selected = Equipement_Choice(vest_Preset, Vest);
        weapon_Selected = Gun_Choice(weapon_Preset, Weapon);

        Instantiate_Equipement(helmet_Selected, helmet_Empty, false);
        Instantiate_Equipement(visor_Selected, visor_Empty, false);
        Instantiate_Equipement(vest_Selected, vest_Empty, false);
        Instantiate_Equipement(weapon_Selected.gun_Prefab, weapon_Empty, true);

        gameObject.GetComponent<IA_Gestion_Equipement>().enabled = false;
    }

    private void Instantiate_Equipement(GameObject o, GameObject emplacement, bool is_Weapon)
    {
        if (o != null)
        {
            GameObject instance = Instantiate(o, emplacement.transform.position, Quaternion.identity, emplacement.transform);
            instance.transform.localRotation = Quaternion.identity;
            if (is_Weapon)
            {
                instance.GetComponent<Gun>().Set_Up(false);
            }
        }
    }

    private GameObject Get_Item_E(GameObject[] o)
    {
        GameObject object_Selected;
        object_Selected = o[Random.Range(0, o.Length)];
        return object_Selected;
    }

    private Gun_Scriptable Get_Item_W(Gun_Scriptable[] w)
    {
        Gun_Scriptable weapon_Selected;
        weapon_Selected = w[Random.Range(0, w.Length)];
        return weapon_Selected;
    }

    private Gun_Scriptable Gun_Choice(Equipement_Type equipement_Preset, WeaponSet w)
    {
        //TODO fix "Nothing" later
        Gun_Scriptable gun_Script_Selected = null;

        Equipement_Type[] values = (Equipement_Type[])System.Enum.GetValues(typeof(Equipement_Type));
        if (equipement_Preset == Equipement_Type.Random)
        {
            equipement_Preset = values[Random.Range(1, 4)];
        }

        switch (equipement_Preset)
        {
            case Equipement_Type.Nothing:
                gun_Script_Selected = null;
                break;
            case Equipement_Type.Light:
                gun_Script_Selected = Get_Item_W(w.Light);
                break;
            case Equipement_Type.Medium:
                gun_Script_Selected = Get_Item_W(w.Medium);
                break;
            case Equipement_Type.Heavy:
                gun_Script_Selected = Get_Item_W(w.Heavy);
                break;
        }
        return gun_Script_Selected;
    }

    private GameObject Equipement_Choice(Equipement_Type equipement_Preset, EquipmentSet e)
    {
        Equipement_Type[] values = (Equipement_Type[])System.Enum.GetValues(typeof(Equipement_Type));
        if (equipement_Preset == Equipement_Type.Random)
        {
            equipement_Preset = values[Random.Range(0, 3)];
        }

        GameObject object_Selected = null;
        switch (equipement_Preset)
        {
            case Equipement_Type.Nothing:
                object_Selected = null;
                break;
            case Equipement_Type.Light:
                object_Selected = Get_Item_E(e.Light);
                break;
            case Equipement_Type.Medium:
                object_Selected = Get_Item_E(e.Medium);
                break;
            case Equipement_Type.Heavy:
                object_Selected = Get_Item_E(e.Heavy);
                break;
        }
        return object_Selected;
    }

    public void Set_Up(Equipement_Type helmet, Equipement_Type visor, Equipement_Type vest, Equipement_Type weapon, GameObject helmet_E, GameObject visor_E, GameObject vest_E, GameObject weapon_E)
    {
        helmet_Empty = helmet_E;
        visor_Empty = visor_E;
        vest_Empty = vest_E;
        weapon_Empty = weapon_E;
        helmet_Preset = helmet;
        vest_Preset = visor;
        visor_Preset = vest;
        weapon_Preset = weapon;
        Equipe_Now();
    }
}
