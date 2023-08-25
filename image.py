from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image

# SMILES表示Achromobacter的某个化学结构
achromobacter_smiles = "CC1=CC(=CC(=C1OC)O)O"

# 将SMILES字符串转换为RDKit的分子对象
molecule = Chem.MolFromSmiles(achromobacter_smiles)

# 绘制分子结构图
image = Draw.MolToImage(molecule, size=(300, 300))

# 设置保存路径
save_path = "D:/graph/hydnocarpin_molecule.png"

# 保存图片到指定路径
image.save(save_path)

print(f"Achromobacter的分子结构图已保存至：{save_path}")
