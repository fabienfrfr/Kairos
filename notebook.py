import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    from transformers import AutoConfig, AutoModel

    # 1. Charger la config officielle
    config = AutoConfig.from_pretrained("google/diffusiongemma-26B-A4B-it")

    # 2. Réduire drastiquement la partie Vision (Gemma4VisionConfig)
    # Réduction de la hidden_size de 1152 à 512 (x2.25) réduit le poids des couches par ~5
    config.vision_config.hidden_size = 512
    config.vision_config.intermediate_size = 2048  # Ratio 4x pour la stabilité
    config.vision_config.num_hidden_layers = 12    # Réduction de profondeur
    config.vision_config.num_attention_heads = 8
    config.vision_config.num_key_value_heads = 8
    config.vision_config.head_dim = 64

    # 3. Appliquer tes modifications texte
    config.text_config.num_hidden_layers = 16
    config.text_config.hidden_size = 512
    config.text_config.intermediate_size = 256
    config.text_config.moe_intermediate_size = 256
    config.text_config.num_experts = 32
    config.text_config.top_k_experts = 4
    config.text_config.num_attention_heads = 8
    config.text_config.num_key_value_heads = 4
    config.text_config.head_dim = 64


    # 2. Forcer le type de couches souhaité
    # On remplace tout par 'sliding_attention'
    config.text_config.layer_types = ["sliding_attention"] * config.text_config.num_hidden_layers

    # 3. Rappel : la classe force la dernière couche en 'full_attention' dans son __post_init__
    # Si tu veux vraiment tout en 'sliding', il faut surcharger cette protection :
    config.text_config.layer_types[-1] = "sliding_attention"

    # 4. Vérification
    print(f"Structure des couches : {config.text_config.layer_types}")


    # 4. Instanciation
    model = AutoModel.from_config(config)

    print("Modèle réduit instancié.")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total des paramètres : {total_params / 1e6:.1f} Millions")
    return (model,)


@app.cell
def _(model):
    model
    return


if __name__ == "__main__":
    app.run()
