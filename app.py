import streamlit as st
from rdkit import Chem
from src.models import GCNModel
import torch
from torch_geometric.data import Data
from src.load_data import get_datasets