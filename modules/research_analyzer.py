from modules.llm import ask_llm


def analyze_paper(text):
    prompt = f"""
You are an advanced AI Research Analyst and Academic Reviewer.

Your task is to deeply analyze the following research paper and generate a detailed, structured, professional academic report.

IMPORTANT INSTRUCTIONS:
- Write in a clean academic style.
- Use proper headings and subheadings.
- Be detailed and analytical, not generic.
- Explain technical concepts in simple but professional language.
- Extract important insights from the paper.
- If some sections are missing, infer intelligently from context.
- Make the response comprehensive and high quality.
- Avoid repeating the same information.
- Use bullet points where appropriate.
- Generate rich and meaningful content.

ANALYZE THE PAPER USING THE FOLLOWING STRUCTURE:

# 1. Paper Overview
- Research domain
- Main objective
- Problem statement
- Why this research is important

# 2. Title Analysis
- Explain the meaning of the title
- What does the title indicate?

# 3. Abstract Summary
- Summarize the abstract in simple language
- Mention the core contribution

# 4. Research Problem
- What problem does the paper try to solve?
- Existing challenges in this field

# 5. Methodology / Proposed Approach
- Explain the methodology step-by-step
- Algorithms used
- Models used
- Frameworks/tools used
- Workflow explanation
- Data processing steps

# 6. Dataset / Data Collection
- Dataset name
- Data source
- Data type
- Number of samples (if available)
- Preprocessing techniques

# 7. System Architecture / Workflow
- Explain the system flow
- Input → Processing → Output
- Important modules/components

# 8. Technologies Used
Mention all technologies, including:
- Programming languages
- Libraries
- Frameworks
- AI/ML models
- APIs
- Databases
- Platforms/tools

# 9. Results and Performance Analysis
- Accuracy/performance metrics
- Observations
- Experimental results
- Comparisons with existing methods
- Key findings

# 10. Advantages / Strengths
Provide detailed strengths of the research.

# 11. Limitations / Weaknesses
Mention realistic limitations of the work.

# 12. Future Scope / Improvements
Suggest advanced future improvements and research directions.

# 13. Real-World Applications
Mention practical applications and industries where this work can be used.

# 14. Innovation and Contribution
- What is novel in this paper?
- Main contribution to research community

# 15. Technical Deep Dive
Explain important technical concepts used in the paper in detail.

# 16. Simplified Explanation
Explain the entire paper in very simple language for beginners/students.

# 17. Interview / Viva Questions
Generate:
- 10 Technical Viva Questions
- 10 Conceptual Questions
- 10 MCQs with answers

# 18. Research Keywords
Extract important technical keywords.

# 19. Suggested Research Paper Title Improvements
Suggest 3 better professional titles for this paper.

# 20. Overall Review
Provide:
- Overall quality assessment
- Research impact
- Practical usefulness
- Final conclusion

RESEARCH PAPER:
{text}
"""

    return ask_llm(prompt, mode="balanced")