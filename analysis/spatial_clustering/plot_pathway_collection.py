import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import csv
 
pathway = 'lipid'
selected_gene_names = [
    "A4GALT", "AACS", "ABCA1", "ABCB11", "ABCB4", "ABCC1", "ABCC3", "ABCD1",
    "ABCG2", "ABHD3", "ABHD4", "ABHD5", "ACAA1", "ACAA2", "ACACA", "ACACB",
    "ACAD10", "ACAD11", "ACADL", "ACADM", "ACADS", "ACADVL", "ACAT1", "ACAT2",
    "ACBD4", "ACBD5", "ACBD6", "ACBD7", "ACER1", "ACER2", "ACER3", "ACHE",
    "ACLY", "ACOT1", "ACOT11", "ACOT12", "ACOT13", "ACOT2", "ACOT4", "ACOT7",
    "ACOT7L", "ACOT8", "ACOT9", "ACOX1", "ACOX2", "ACOX3", "ACOXL", "ACP6",
    "ACSBG1", "ACSBG2", "ACSF2", "ACSF3", "ACSL1", "ACSL3", "ACSL4", "ACSL5",
    "ACSL6", "ACSM3", "ACSM6", "ACSS3", "AGK", "AGMO", "AGPAT1", "AGPAT2",
    "AGPAT3", "AGPAT4", "AGPAT5", "AGPS", "AGT", "AHR", "AHRR", "AKR1B1",
    "AKR1B15", "AKR1C1", "AKR1C2", "AKR1C3", "AKR1C4", "AKR1D1", "ALAS1",
    "ALB", "ALDH3A2", "ALDH3B1", "ALDH3B2", "ALOX12", "ALOX12B", "ALOX15",
    "ALOX15B", "ALOX5", "ALOX5AP", "ALOXE3", "ALPI", "AMACR", "ANGPTL4",
    "ANKRD1", "APOA1", "APOA2", "APOA5", "ARF1", "ARF3", "ARNT", "ARNT2",
    "ARSA", "ARSB", "ARSD", "ARSF", "ARSG", "ARSH", "ARSI", "ARSJ", "ARSK",
    "ARSL", "ARV1", "ASAH1", "ASAH2", "AWAT1", "AWAT2", "B3GALNT1", "B3GALT4",
    "B3GNT5", "B4GALNT1", "B4GALT5", "B4GALT6", "BAAT", "BCHE", "BDH1",
    "BDH2", "BMAL1", "BMX", "CARM1", "CAV1", "CBR1", "CBR4", "CCNC", "CD36",
    "CDIPT", "CDK19", "CDK8", "CDS1", "CDS2", "CEPT1", "CERK", "CERS1",
    "CERS2", "CERS3", "CERS4", "CERS5", "CERS6", "CERT1", "CGA", "CH25H",
    "CHAT", "CHD9", "CHKA", "CHKB", "CHPT1", "CIDEA", "CIDEC", "CLOCK",
    "CPNE1", "CPNE3", "CPNE6", "CPNE7", "CPT1A", "CPT1B", "CPT2", "CRAT",
    "CREBBP", "CRLS1", "CROT", "CSNK1G2", "CSNK2A1", "CSNK2A2", "CSNK2B",
    "CTSA", "CUBN", "CYB5B", "CYP11A1", "CYP11B1", "CYP11B2", "CYP17A1",
    "CYP19A1", "CYP1A1", "CYP1A2", "CYP1B1", "CYP21A2", "CYP24A1", "CYP27A1",
    "CYP27B1", "CYP2C19", "CYP2C8", "CYP2C9", "CYP2D6", "CYP2E1", "CYP2J2",    
    "CYP2R1", "CYP2U1", "CYP39A1", "CYP3A4", "CYP46A1", "CYP4A11", "CYP4A22",
    "CYP4B1", "CYP4F11", "CYP4F2", "CYP4F22", "CYP4F3", "CYP4F8", "CYP51A1",
    "CYP7A1", "CYP7B1", "CYP8B1", "DBI", "DDHD1", "DDHD2", "DECR1", "DECR2",
    "DEGS1", "DEGS2", "DGAT1", "DGAT2", "DGAT2L6", "DHCR24", "DHCR7",
    "DHRS7B", "DPEP1", "DPEP2", "EBP", "ECHS1", "ECI1", "ECI2", "EHHADH",
    "ELOVL1", "ELOVL2", "ELOVL3", "ELOVL4", "ELOVL5", "ELOVL6", "ELOVL7",
    "ENPP6", "ENPP7", "EP300", "EPHX2", "ESRRA", "ETNK1", "ETNK2", "ETNPPL",
    "FA2H", "FAAH", "FAAH2", "FABP1", "FABP12", "FABP2", "FABP3", "FABP4",  
    "FABP5", "FABP6", "FABP7", "FABP9", "FADS1", "FADS2", "FAM120B", "FAR1",
    "FAR2", "FASN", "FDFT1", "FDPS", "FDX1", "FDX2", "FDXR", "FHL2", "FIG4",
    "FITM1", "FITM2", "FUT1", "FUT2", "G0S2", "GAL3ST1", "GALC", "GBA1",
    "GBA2", "GBA3", "GC", "GDE1", "GDPD1", "GDPD3", "GDPD5", "GGPS1",
    "GGT1", "GGT5", "GK", "GK2", "GK3", "GLA", "GLB1", "GLB1L", "GLB1L2",
    "GLB1L3", "GLIPR1", "GM2A", "GNPAT", "GPAM", "GPAT2", "GPAT3", "GPAT4",
    "GPCPD1", "GPD1", "GPD1L", "GPD2", "GPS2", "GPX1", "GPX2", "GPX4",
    "GRHL1", "GSTM4", "HACD1", "HACD2", "HACD3", "HACD4", "HACL1", "HADH",
    "HADHA", "HADHB", "HAO2", "HDAC3", "HELZ2", "HEXA", "HEXB", "HILPDA",
    "HMGCL", "HMGCLL1", "HMGCR", "HMGCS1", "HMGCS2", "HPGD", "HPGDS",
    "HSD11B1", "HSD11B2", "HSD17B1", "HSD17B11", "HSD17B12", "HSD17B13",
    "HSD17B14", "HSD17B2", "HSD17B3", "HSD17B4", "HSD17B7", "HSD17B8",
    "HSD3B1", "HSD3B2", "HSD3B7", "HTD2", "IDI1", "IDI2", "INPP4A", "INPP4B",
    "INPP5D", "INPP5E", "INPP5F", "INPP5J", "INPP5K", "INPPL1", "INSIG1",
    "INSIG2", "KDSR", "KPNB1", "LBR", "LCLAT1", "LDLRAP1", "LGMN", "LHB",
    "LIPE", "LIPH", "LIPI", "LPCAT1", "LPCAT2", "LPCAT3", "LPCAT4", "LPGAT1",
    "LPIN1", "LPIN2", "LPIN3", "LRP2", "LSS", "LTA4H", "LTC4S", "M6PR",
    "MAPKAPK2", "MBOAT1", "MBOAT2", "MBOAT7", "MBTPS1", "MBTPS2", "MCAT",
    "MCEE", "ME1", "MECR", "MED1", "MED10", "MED11", "MED12", "MED13",
    "MED13L", "MED14", "MED15", "MED16", "MED17", "MED18", "MED19", "MED20",
    "MED21", "MED22", "MED23", "MED24", "MED25", "MED26", "MED27", "MED28",
    "MED29", "MED30", "MED31", "MED4", "MED6", "MED7", "MED8", "MED9",
    "MFSD2A", "MFSD2B", "MGLL", "MID1IP1", "MIGA1", "MIGA2", "MLYCD",
    "MMAA", "MMUT", "MOGAT1", "MOGAT2", "MOGAT3", "MORC2", "MSMO1", "MTF1",
    "MTM1", "MTMR1", "MTMR10", "MTMR12", "MTMR14", "MTMR2", "MTMR3",
    "MTMR4", "MTMR6", "MTMR7", "MTMR8", "MTMR9", "MVD", "MVK", "NCOA1",
    "NCOA2", "NCOA3", "NCOA6", "NCOR1", "NCOR2", "NDUFAB1", "NEU1", "NEU2",
    "NEU3", "NEU4", "NFYA", "NFYB", "NFYC", "NPAS2", "NR1D1", "NR1H2",
    "NR1H3", "NR1H4", "NRF1", "NSDHL", "NUDT19", "NUDT7", "OCRL", "OLAH",
    "ORMDL1", "ORMDL2", "ORMDL3", "OSBP", "OSBPL10", "OSBPL1A", "OSBPL2",
    "OSBPL3", "OSBPL5", "OSBPL6", "OSBPL7", "OSBPL8", "OSBPL9", "OXCT1",
    "OXCT2", "PCCA", "PCCB", "PCTP", "PCYT1A", "PCYT1B", "PCYT2", "PECR",
    "PEMT", "PEX11A", "PGP", "PGS1", "PHOSPHO1", "PHYH", "PI4K2A", "PI4K2B",
    "PI4KA", "PI4KB", "PIAS4", "PIK3C2A", "PIK3C2B", "PIK3C2G", "PIK3C3",
    "PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG", "PIK3R1", "PIK3R2", "PIK3R3",
    "PIK3R4", "PIK3R5", "PIK3R6", "PIKFYVE", "PIP4K2A", "PIP4K2B", "PIP4K2C",
    "PIP4P1", "PIP5K1A", "PIP5K1B", "PIP5K1C", "PISD", "PITPNB", "PITPNM1",
    "PITPNM2", "PITPNM3", "PLA1A", "PLA2G10", "PLA2G12A", "PLA2G15",
    "PLA2G1B", "PLA2G2A", "PLA2G2D", "PLA2G2E", "PLA2G2F", "PLA2G3",
    "PLA2G4A", "PLA2G4B", "PLA2G4C", "PLA2G4D", "PLA2G4E", "PLA2G4F",
    "PLA2G5", "PLA2G6", "PLA2R1", "PLAAT1", "PLAAT2", "PLAAT3", "PLAAT4",
    "PLAAT5", "PLB1", "PLBD1", "PLD1", "PLD2", "PLD3", "PLD4", "PLD6",
    "PLEKHA1", "PLEKHA2", "PLEKHA3", "PLEKHA4", "PLEKHA5", "PLEKHA6",
    "PLEKHA8", "PLIN1", "PLIN2", "PLIN3", "PLPP1", "PLPP2", "PLPP3", "PLPP6",
    "PMVK", "PNPLA2", "PNPLA3", "PNPLA4", "PNPLA5", "PNPLA6", "PNPLA7",
    "PNPLA8", "POMC", "PON1", "PON2", "PON3", "PPARA", "PPARD", "PPARG",
    "PPARGC1A", "PPARGC1B", "PPM1L", "PPP1CA", "PPP1CB", "PPP1CC", "PPT1",
    "PPT2", "PRKAA2", "PRKAB2", "PRKACA", "PRKACB", "PRKACG", "PRKAG2",
    "PRKD1", "PRKD2", "PRKD3", "PRXL2B", "PSAP", "PTDSS1", "PTDSS2", "PTEN",
    "PTGDS", "PTGES", "PTGES2", "PTGES3", "PTGIS", "PTGR1", "PTGR2", "PTGS1",
    "PTGS2", "PTPMT1", "PTPN13", "RAB14", "RAB4A", "RAB5A", "RAN", "RGL1",
    "RORA", "RUFY1", "RXRA", "RXRB", "SACM1L", "SAMD8", "SAR1B", "SBF1",
    "SBF2", "SC5D", "SCAP", "SCD", "SCD5", "SCP2", "SEC23A", "SEC24A",
    "SEC24B", "SEC24C", "SEC24D", "SELENOI", "SERPINA6", "SGMS1", "SGMS2",
    "SGPL1", "SGPP1", "SGPP2", "SIN3A", "SIN3B", "SLC10A1", "SLC10A2",
    "SLC22A5", "SLC25A17", "SLC25A20", "SLC27A1", "SLC27A2", "SLC27A3",
    "SLC27A5", "SLC44A1", "SLC44A2", "SLC44A3", "SLC44A4", "SLC44A5",
    "SLC51A", "SLC51B", "SLCO1A2", "SLCO1B1", "SLCO1B3", "SMARCD3", "SMPD1",
    "SMPD2", "SMPD3", "SMPD4", "SP1", "SPHK1", "SPHK2", "SPNS2", "SPTLC1",
    "SPTLC2", "SPTLC3", "SPTSSA", "SPTSSB", "SQLE", "SRD5A1", "SRD5A2",
    "SRD5A3", "SREBF1", "SREBF2", "ST3GAL2", "ST3GAL3", "ST3GAL5",
    "ST6GALNAC5", "ST6GALNAC6", "ST8SIA5", "STAR", "STARD10", "STARD3",
    "STARD3NL", "STARD4", "STARD5", "STARD6", "STARD7", "STS", "SULT2A1",
    "SUMF1", "SUMF2", "SUMO2", "SYNJ1", "SYNJ2", "TAFAZZIN", "TBL1X",
    "TBL1XR1", "TBXAS1", "TECR", "TECRL", "TGS1", "THEM4", "THEM5", "THRAP3",
    "THRSP", "TIAM2", "TM7SF2", "TMEM86B", "TNFAIP8", "TNFAIP8L1",
    "TNFAIP8L2", "TNFAIP8L3", "TNFRSF21", "TPTE", "TPTE2", "TRIB3", "TSPO",
    "TSPOAP1", "TXNRD1", "UBE2I", "UGCG", "UGT1A9", "UGT8", "VAC14", "VAPA",
    "VAPB", "VDR"
]

print(len(selected_gene_names))

gene_order  = pickle.load(open('protein_order.pkl', 'rb'))
indices     = []
valid_genes = []
for name in selected_gene_names:
    for idx, g in enumerate(gene_order):
        if name.lower() in g.lower():
            indices.append(idx)
            valid_genes.append(name)
            break
    else:
        print(f"  Gene {name} not found in gene_order — skipping")

print(f"Matched {len(valid_genes)} / {len(selected_gene_names)} genes.")


IMAGE_DIR  = '/scratch/users/sunh/cytoVL'
cell_line  = 'Hep-G2'
j          = 9
stack_path = f'{IMAGE_DIR}/{cell_line}_superplexed_{j:03d}_stacked_images.npy'

img = np.load(stack_path)
sub = img[:, :, indices].astype(np.float32)   # (H, W, n_genes)

# ── Layout ────────────────────────────────────────────────────────────────────

N_COLS     = 26
COL_LABELS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
n_genes    = len(valid_genes)
N_ROWS     = min(int(np.ceil(n_genes / N_COLS)), 40)

PANEL_W    = 0.32      # inches per panel
PANEL_H    = 0.32      # inches per panel
H_PAD      = 0.02      # inches between rows
W_PAD      = 0.02      # inches between columns
MARGIN_L   = 0.25      # left margin for row labels
MARGIN_T   = 0.18      # top margin for column labels

fig_w = MARGIN_L + N_COLS * PANEL_W + (N_COLS - 1) * W_PAD
fig_h = MARGIN_T + N_ROWS * PANEL_H + (N_ROWS - 1) * H_PAD

fig, axes = plt.subplots(
    N_ROWS, N_COLS,
    figsize=(fig_w, fig_h),
    squeeze=False,
)
fig.subplots_adjust(
    left   = MARGIN_L / fig_w,
    right  = 1.0,
    top    = 1.0 - MARGIN_T / fig_h,
    bottom = 0.0,
    wspace = W_PAD / PANEL_W,
    hspace = H_PAD / PANEL_H,
)

# ── Scale bar ─────────────────────────────────────────────────────────────────
SCALE_BAR_UM     = 20
SCALE_BAR_PX     = 93.721 * 2      # 20 µm / 0.1067 µm·px⁻¹ ≈ 187.44 px
SCALE_BAR_COLOR  = 'white'
SCALE_BAR_HEIGHT = 15               # bar thickness in native pixels
SCALE_BAR_MARGIN = 18               # distance from bottom-left corner in native pixels

def draw_scale_bar(ax, img_h=512, img_w=512):
    """Draw a 20 µm scale bar in the bottom-left corner (data coords = native px)."""
    from matplotlib.patches import Rectangle
    x0 = SCALE_BAR_MARGIN
    y0 = img_h - SCALE_BAR_MARGIN - SCALE_BAR_HEIGHT
    ax.add_patch(Rectangle(
        (x0, y0), SCALE_BAR_PX, SCALE_BAR_HEIGHT,
        color=SCALE_BAR_COLOR, transform=ax.transData, clip_on=True,
    ))

green_cmap = mcolors.LinearSegmentedColormap.from_list(
    'black_green', [(0, 0, 0), (0, 1, 0)]
)

for idx in range(N_ROWS * N_COLS):
    row, col = divmod(idx, N_COLS)
    ax       = axes[row][col]

    # ── column header (top row only) ──────────────────────────────────────────
    if row == 0:
        ax.set_title(COL_LABELS[col], fontsize=5, pad=1.5, color='black')

    # ── row number (left column only) ─────────────────────────────────────────
    if col == 0:
        ax.set_ylabel(str(row + 1), fontsize=5, labelpad=2,
                      rotation=0, va='center', ha='right')

    if idx >= n_genes:
        ax.axis('off')
        continue

    channel = sub[:, :, idx]
    lo, hi  = np.percentile(channel, 1), np.percentile(channel, 99.9)
    channel_disp = np.clip((channel - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(channel)

    ax.imshow(channel_disp, cmap=green_cmap, vmin=0, vmax=1,
              interpolation='nearest')

    # ── Scale bar on first panel only ─────────────────────────────────────────
    if idx == 0:
        draw_scale_bar(ax, img_h=sub.shape[0], img_w=sub.shape[1])

    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)

out_path = f'{cell_line}_j{j:03d}_{pathway}_atlas.png'
fig.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
print(f"Saved: {out_path}")
plt.show()

# ── Save gene name layout to CSV ──────────────────────────────────────────────

grid = [[''] * N_COLS for _ in range(N_ROWS)]
for idx, gene in enumerate(valid_genes):
    row, col      = divmod(idx, N_COLS)
    grid[row][col] = gene

csv_path = f'{cell_line}_j{j:03d}_{pathway}_atlas.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(COL_LABELS)
    for row_data in grid:
        writer.writerow(row_data)
print(f"Saved: {csv_path}")