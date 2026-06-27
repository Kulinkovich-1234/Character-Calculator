# 特征标计算器 (Character Table Decomposer)

一个用于点群特征标计算的交互式 Python 程序，内置 **42 种常见点群** 的特征标表，
支持张量积、对称积、反对称积、球谐函数、多项式表示、振动模式分析等多种操作，
并能将任意可约表示分解为不可约表示的直和。同时提供**表达式引擎**和**命令行（CLI）模式**，
支持从简单一行命令执行计算，无需进入交互菜单。

---

## 📦 版本信息
- **版本**：3.0.0
- **作者**：Jianwen Ma
- **日期**：2026-06-28
- **许可证**：[MIT License](LICENSE)
- **版权**：Copyright © 2026 Jianwen Ma

---

## ✨ 主要特性

### 群论计算
- **内置 42 个点群**：覆盖从低对称性到二十面体群的全部类别：
  $C_1, C_s, C_i, C_n, C_{nv}, C_{nh}, D_n, D_{nh}, D_{nd}, S_n, T, T_h, T_d, O, O_h$, **$I$, $I_h$**（新增）
- **可约表示分解**：输入任意特征标向量，自动分解为不可约表示的直和
- **张量积与直和**：$\chi_1 \otimes \chi_2$、$\chi_1 \oplus \chi_2$
- **对称积与反对称积**：$\mathrm{Sym}^n(\chi)$、$\mathrm{Alt}^n(\chi)$，基于置换群共轭类公式
- **球谐函数表示**：按角量子数 $l$ 或轨道字母（s, p, d, f…）生成原子轨道特征标
- **多项式表示**：$\mathrm{Sym}^n(\mathbf{V})$，即 $n$ 次齐次多项式空间
- **幂次特征标**：$\chi(g^n)$ 与张量幂 $\chi^{\otimes n}$

### 表达式引擎（v3.0.0 新增）
- 支持**中缀表达式**，一行命令完成多步运算
- 语法：`Sym^2([T1u]) x [Eg] ⊕ Alt^2([T2g])`
- 所有特征标引用通过 `[...]` 包裹，完全消除运算符歧义
- 支持手动输入 `[3, 0, -1, 1]`、不可约表示引用 `[T1u]`、向量表示 `[Vec]`
- UTF-8 Unicode 运算符（`⊗` `⊕`）与 ASCII（`x` `+`）兼容

### 振动模式分析（v3.0.0 新增）
- 基于群论的分子振动分析
- 输入每个对称操作下保持不动的原子数，自动计算：
  - $\Gamma_{\text{total}}$、$\Gamma_{\text{trans}}$、$\Gamma_{\text{rot}}$、$\Gamma_{\text{vib}}$ 的不可约表示分解
  - 红外和拉曼活性振动模及其重数

### CLI 模式（v3.0.0 新增）
- 支持算式求值 `-g O_h "Sym^2([T1u])"`
- 支持光谱活性查询 `-g O_h IR` / `-g O_h Raman`
- 支持振动分析 `-g C_3v vib 4 1 2`
- 结果保存 `-> name` 与自动命名
- JSON 输出 `--json` 便于脚本集成
- 支持存储管理 `storage --group O_h list`

### 旧版交互模式（完全保留）
- 所有现有菜单界面、自定义存储、表验证等功能不变

### 验证增强（v3.0.0 新增）
- 新增张量积乘法表检查
- 新增对称/反对称积 $\mathrm{Sym}^{2-5}$、$\mathrm{Alt}^{2-5}$ 整数性检查

---

## 🔧 安装与运行

### 环境要求
- Python 3.6 或更高版本
- NumPy（用于复数运算和向量处理）

### 安装步骤
1. 克隆本仓库到本地：
   ```bash
   git clone https://github.com/Kulinkovich-1234/Character-Calculator.git
   cd Character-Calculator
   ```

2. （可选）创建并激活虚拟环境：
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. 安装依赖：
   ```bash
   pip install numpy
   ```

4. 运行程序（交互模式）：
   ```bash
   python main.py
   ```

---

## 🚀 使用指南

### 交互模式
运行后显示按分类排序的点群列表，输入编号进入操作界面。
详细用法见原交互菜单。

### CLI 模式
```bash
python main.py list                            # 列出所有点群
python main.py -g O_h "Sym^2([T1u])"          # 对称积
python main.py -g O_h "[T1u] x [Eg]"          # 张量积
python main.py -g O_h IR                      # 红外活性
python main.py -g O_h Raman                   # 拉曼活性
python main.py -g O_h table                   # 特征标表
python main.py -g C_3v vib 4 1 2             # 振动分析
python main.py -g I_h "Y(2)"                 # 球谐函数
python main.py verify --all                   # 验证所有群
python main.py -g O_h "Sym^2([Vec]) -> r"    # 保存结果
```

完整命令及示例见 [CLI_USAGE.md](CLI_USAGE.md)。

---

## 📚 内置点群列表

程序包含以下 42 个点群的特征标表（按分类排序）：

| 分类          | 点群                                                                 |
|---------------|----------------------------------------------------------------------|
| Nonaxial      | $C_1, C_s$                                                         |
| $C_n$         | $C_2, C_3, C_4, C_5, C_6$                                          |
| $C_{nv}$      | $C_{2v}, C_{3v}, C_{4v}, C_{5v}, C_{6v}$                           |
| $C_{nh}$      | $C_{2h}, C_{3h}, C_{4h}, C_{5h}, C_{6h}$                           |
| $D_n$         | $D_2, D_3, D_4, D_5, D_6$                                          |
| $D_{nh}$      | $D_{2h}, D_{3h}, D_{4h}, D_{5h}, D_{6h}$                           |
| $D_{nd}$      | $D_{2d}, D_{3d}, D_{4d}, D_{5d}, D_{6d}$                           |
| $S_n$         | $C_i (=S_2), S_4, S_6$                                             |
| Cubic         | $T, T_h, T_d, O, O_h$                                              |
| **I（新增）**   | **$I, I_h$**                                                      |

所有表均已通过增强验证（8 项检查，含张量积乘法表和对称/反对称积）。

---

## 📄 许可证
本项目采用 MIT 许可证。您可以自由使用、修改和分发本软件，但需保留原始版权声明。详见 [LICENSE](LICENSE) 文件。

---

## 👤 作者
- **Jianwen Ma** – 开发和维护
- 如有问题或建议，欢迎在 GitHub 上提交 Issue 或 Pull Request。

---

## 🙏 致谢
感谢所有群表示论教材和参考资料，本程序的特征标表数据主要参考了经典的群论书籍和在线资源，并经过严格验证。

---

**尽情探索点群的表示世界吧！**
