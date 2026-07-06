import torch
import ast

def main():
        
    data = torch.load("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/PINNACLE/pinnacle_embeds/pinnacle_embeds/pinnacle_protein_embed.pth")
    print(type(data))
    if isinstance(data, dict):
        print(data.keys())
        for k, v in data.items():
            print(f"{k}: {type(v)}, {v.shape if hasattr(v, 'shape') else v}")
    else:
        print(data.shape)
    
if __name__ == "__main__":
    main()