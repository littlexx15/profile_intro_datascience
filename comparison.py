import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load images (please update the paths as necessary)
img_white_without = mpimg.imread("/Users/xiangxiaoxin/Documents/GitHub/profile_intro_datascience/week2-APIs/image/search_white_without_k.jpg")
img_white_with    = mpimg.imread("/Users/xiangxiaoxin/Documents/GitHub/profile_intro_datascience/week2-APIs/image/search_white_with_k.jpg")
img_red_without   = mpimg.imread("/Users/xiangxiaoxin/Documents/GitHub/profile_intro_datascience/week2-APIs/image/search_red_without_k.jpg")
img_red_with      = mpimg.imread("/Users/xiangxiaoxin/Documents/GitHub/profile_intro_datascience/week2-APIs/image/search_red_with_k.jpg")

# Create a figure with 2 rows and 2 columns of subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Function to add a thin black border around each subplot
def add_border(ax, color='black', linewidth=2):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(linewidth)

# ----- Row 1: "搜索白色" -----

# Left: Without K-Means
axes[0, 0].imshow(img_white_without)
axes[0, 0].axis('off')  # Hide axis ticks and labels
axes[0, 0].set_title("搜索白色 - 不使用 K-Means", fontsize=14, fontweight='bold')
add_border(axes[0, 0])

# Right: With K-Means (k=3)
axes[0, 1].imshow(img_white_with)
axes[0, 1].axis('off')
axes[0, 1].set_title("搜索白色 - 使用 K-Means (k=3)", fontsize=14, fontweight='bold')
add_border(axes[0, 1])

# ----- Row 2: "搜索红色" -----

# Left: Without K-Means
axes[1, 0].imshow(img_red_without)
axes[1, 0].axis('off')
axes[1, 0].set_title("搜索红色 - 不使用 K-Means", fontsize=14, fontweight='bold')
add_border(axes[1, 0])

# Right: With K-Means (k=3)
axes[1, 1].imshow(img_red_with)
axes[1, 1].axis('off')
axes[1, 1].set_title("搜索红色 - 使用 K-Means (k=3)", fontsize=14, fontweight='bold')
add_border(axes[1, 1])

plt.tight_layout()
plt.show()
