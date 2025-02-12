import matplotlib.pyplot as plt
# Re-create the plot with adjustments to avoid overlap between "Danish Environment Agency" and "Danish Road Directorate".

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Matrix sections for visual aid
plt.axvline(x=5, color="grey", linestyle="--")
plt.axhline(y=5, color="grey", linestyle="--")

# Set labels for the axis without showing tick numbers
ax.set_xlabel("Influence", fontsize=14)
ax.set_ylabel("Complicity", fontsize=14)
ax.set_xticks([])
ax.set_yticks([])
# ax.set_title("Stakeholder Influence and Complicity Matrix on Highway Traffic", fontsize=16)

# Adjusted positions to avoid overlap
stakeholders = [
    (1, 3.5, "Security", "center", "center"),
    (1, 4, "Investors", "center", "center"),
    (1, 4.5, "Subjects", "center", "center"),
    (1, 6, "Insurance", "center", "center"),
    (1.2, 5.5, "Competitors", "center", "center"),
    (1.7, 6.5, "Emergency Services", "center", "center"),
    (4, 9, "FoodHub", "center", "center"),
    (6, 3.5, "Customers", "center", "center"),
    (8.1, 2, "Data Protection Agency", "center", "center"),
    (7.6, 1.5, "Working Environment Authority", "center", "center"),
    (9, 9, "INNOVATE", "center", "center"),
]

# Plot each stakeholder in the matrix
for x, y, label, ha, va in stakeholders:
    ax.text(x, y, label, ha=ha, va=va, fontsize=12,
            bbox=dict(facecolor="lightblue", edgecolor="black", boxstyle="round,pad=0.3"))

# Label the quadrants
ax.text(2.5, 9.5, "High Complicity\nLow Influence", ha="center", va="center", fontsize=12, color="grey")
ax.text(7.5, 9.5, "High Complicity\nHigh Influence", ha="center", va="center", fontsize=12, color="grey")
ax.text(2.5, 0.5, "Low Complicity\nLow Influence", ha="center", va="center", fontsize=12, color="grey")
ax.text(7.5, 0.5, "Low Complicity\nHigh Influence", ha="center", va="center", fontsize=12, color="grey")

# Display the plot
plt.grid(False)
plt.show()
