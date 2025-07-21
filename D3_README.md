## 📊 **D3.js Parliamentary Data Visualization**

### **1. Basic Metrics Dashboard**
The visualization starts with key dataset metrics displayed in an attractive card layout:
- Total Questions count
- Number of unique Departments
- Number of unique Deputies  
- Number of unique Question Types
- Average Question Length
- Date Range of the dataset

### **2. Hierarchical Bubble Chart**
A sophisticated interactive bubble chart with three different views:

**By Department View**: 
- Large bubbles = government departments
- Small bubbles inside = deputies within each department
- Size indicates question volume
- Color-coded by department

**By Deputy View**:
- Large bubbles = individual deputies
- Small bubbles inside = departments they've questioned
- Shows cross-departmental activity patterns

**By Question Type View**:
- Large bubbles = question categories/headings
- Small bubbles inside = departments for each question type
- Reveals which departments are most questioned on specific topics

### **Interactive Features**
- **Hover tooltips** with detailed counts and percentages
- **View switching** buttons to toggle between different perspectives
- **Responsive design** that adapts to screen size
- **Smooth animations** and transitions

### **How to Run**
```bash
python serve_d3.py
```

This will automatically start a local server and open the visualization in your browser. The visualization will load your `parliamentary_data_clean.csv` file and display the interactive charts.

The D3.js implementation uses modern web technologies with a beautiful gradient design, smooth interactions, and professional styling that makes the parliamentary data easy to explore and understand. 