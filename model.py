"""
model.py
--------
Arsitektur Model Late Fusion Multimodal untuk Automated Essay Scoring.
Menggabungkan fitur Visual (ResNet-50) dan fitur Teks (DistilBERT).
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel
from torchvision.models import resnet50, ResNet50_Weights

class MultimodalEssayModel(nn.Module):
    def __init__(self, fusion_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        
        # 1. Cabang Teks: DistilBERT
        # Mengambil model dasar pretrained
        self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        text_features_dim = self.text_model.config.hidden_size # Default: 768
        
        # 2. Cabang Gambar: ResNet-50
        # Menggunakan weights standar ImageNet v2
        weights = ResNet50_Weights.DEFAULT
        self.vision_model = resnet50(weights=weights)
        vision_features_dim = self.vision_model.fc.in_features # Default: 2048
        
        # Potong layer FC (klasifikasi) bawaan ResNet agar kita mendapat ekstrak fitur (pooling)
        self.vision_model.fc = nn.Identity()
        
        # 3. Lapisan Penggabungan (Late Fusion Via Concatenation)
        total_multimodal_dim = text_features_dim + vision_features_dim
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_multimodal_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        
        # 4. Output Layer (Regressor ke Skala [0, 1])
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid() # Memastikan output mutlak berada di rentang 0 sampai 1
        )

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Extractor Teks: Mengambil representasi token [CLS] (indeks 0)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :] # Shape: [batch_size, 768]
        
        # Extractor Gambar
        vision_features = self.vision_model(image) # Shape: [batch_size, 2048]
        
        # Konkatensi Fitur Teks + Gambar
        combined_features = torch.cat((text_features, vision_features), dim=1) # Shape: [batch_size, 2816]
        
        # Proses Fusion & Reduksi Dimensi
        fused_vector = self.fusion_layer(combined_features)
        
        # Prediksi Skor
        output_score = self.regressor(fused_vector)
        
        return output_score.squeeze(-1) # Kembalikan shape tensor menjadi [batch_size] agar cocok dengan MSELoss