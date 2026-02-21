---
name: todoist-tasks
description: Create Todoist tasks from natural language. Use when the user mentions a task, to-do, reminder, or something they need to do. Triggers on phrases like "I need to", "remind me to", "add task", or any actionable item the user wants to track.
---

# Todoist Task Creation

Create tasks in Todoist from natural language input using the MCP Todoist tools.

## User's Todoist Structure

**All tasks go to Inbox** — no separate projects.

**Labels for categorization:**
- `professional` — PhD hunting, job hunting, networking with faculty/professors, LinkedIn, career counseling, industry networking
- `social` — social activities, friends, dating, community

**Active vs Backlog (determined by due date):**
- Due today = active
- Due on other days or no due date = backlog

## Defaults

| Field | Default | Override when... |
|-------|---------|------------------|
| Project | Inbox | Always Inbox |
| Label | None | Content matches professional or social domain |
| Priority | p4 (lowest) | User says "high priority", "important", "urgent" |
| Due date | None (backlog) | User specifies "today", "tomorrow", a date, etc. |

## Label Auto-Detection

Infer label from task content:

| Keywords/Context | Label |
|------------------|-------|
| faculty, professor, PhD, grad school, research, academia, department, advisor, LinkedIn, recruiter, job, career, industry, interview, resume, networking (professional) | professional |
| friends, hangout, party, dinner, drinks, social event, birthday, dating, Hinge, community | social |
| Everything else | no label |

## Workflow

1. Parse the user's natural language for: task content, label hints, priority, due date
2. Apply defaults for any unspecified fields
3. Call `mcp__todoist__add-tasks` with the parsed values
4. Confirm creation with task name and key details

## MCP Tool Reference

```
mcp__todoist__add-tasks
Parameters:
- tasks: array of task objects
  - content: string (required) — task title
  - description: string — additional details
  - labels: array of strings — e.g., ["professional"] or ["social"]
  - priority: "p1" | "p2" | "p3" | "p4" — p1 highest, p4 lowest
  - dueString: string — natural language date like "tomorrow", "next Friday"
  - duration: string — e.g., "2h", "30m"
```

## Examples

**User:** "I need to email Professor Smith about research opportunities"
→ Inbox, Label: professional, Priority: p4, No due date

**User:** "Update my LinkedIn profile today"
→ Inbox, Label: professional, Priority: p4, Due: today

**User:** "Get groceries tomorrow"
→ Inbox, No label, Priority: p4, Due: tomorrow

**User:** "High priority: prepare for interview on Friday"
→ Inbox, Label: professional, Priority: p1, Due: Friday

**User:** "Grab coffee with Sarah this weekend"
→ Inbox, Label: social, Priority: p4, Due: this weekend
