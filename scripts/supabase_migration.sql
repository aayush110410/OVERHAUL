-- Supabase Migration: Customer Validation Table
-- Run this in the Supabase SQL Editor to create the validation_entries table

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create the validation_entries table
CREATE TABLE IF NOT EXISTS validation_entries (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    
    -- Basic info
    name VARCHAR(100) NOT NULL DEFAULT 'Anonymous',
    email VARCHAR(254) NOT NULL,
    role VARCHAR(80),  -- Student, Faculty/Researcher, Professional, Other
    organization VARCHAR(120),
    
    -- Q1: Problem relevance (1-5)
    problem_relevance INTEGER NOT NULL CHECK (problem_relevance >= 1 AND problem_relevance <= 5),
    
    -- Q2: Personal experience with challenges (yes/no)
    has_experience BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Q3: Where current tools fall short
    tools_shortcoming TEXT,
    
    -- Q4: Context for OVERHAUL (multiple select, stored as JSON array)
    usage_contexts JSONB DEFAULT '[]'::jsonb,
    
    -- Q5: Value as learning tool (yes/maybe/no)
    learning_tool_value VARCHAR(10),
    
    -- Q6: Most interesting aspect (optional)
    interesting_aspect VARCHAR(100),
    
    -- Q7: Suggested improvement (optional)
    suggested_improvement TEXT,
    
    -- Q8: Interest in trying (optional)
    interest_in_trying VARCHAR(10),
    
    -- Legacy fields for backwards compatibility
    rating INTEGER DEFAULT 5,
    feedback TEXT,
    need_validation TEXT,
    intended_use TEXT,
    
    -- Location (optional)
    location_query VARCHAR(200),
    location_label VARCHAR(300),
    location_lat VARCHAR(40),
    location_lon VARCHAR(40),
    
    -- Moderation
    is_approved BOOLEAN NOT NULL DEFAULT TRUE,
    moderation_reason VARCHAR(200)
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_validation_created_at ON validation_entries(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_validation_approved ON validation_entries(is_approved);
CREATE INDEX IF NOT EXISTS idx_validation_email ON validation_entries(email);

-- Enable Row Level Security (recommended for Supabase)
ALTER TABLE validation_entries ENABLE ROW LEVEL SECURITY;

-- Create policy for public read access (approved entries only)
CREATE POLICY "Public can view approved entries" ON validation_entries
    FOR SELECT
    USING (is_approved = TRUE);

-- Create policy for authenticated insert
CREATE POLICY "Anyone can insert entries" ON validation_entries
    FOR INSERT
    WITH CHECK (TRUE);

-- Optional: Create a view for the live feed (recent entries)
CREATE OR REPLACE VIEW recent_validation_entries AS
SELECT 
    id,
    created_at,
    name,
    CONCAT(LEFT(SPLIT_PART(email, '@', 1), 1), '***@', SPLIT_PART(email, '@', 2)) as display_email,
    role,
    organization,
    problem_relevance,
    has_experience,
    tools_shortcoming,
    usage_contexts,
    learning_tool_value,
    interesting_aspect,
    suggested_improvement,
    interest_in_trying
FROM validation_entries
WHERE is_approved = TRUE
ORDER BY created_at DESC
LIMIT 50;

-- Grant access to the view
GRANT SELECT ON recent_validation_entries TO anon;
GRANT SELECT ON recent_validation_entries TO authenticated;

-- Stats function for the live counter
CREATE OR REPLACE FUNCTION get_validation_stats()
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'total_entries', COUNT(*),
        'approved_entries', COUNT(*) FILTER (WHERE is_approved = TRUE),
        'avg_relevance', ROUND(AVG(problem_relevance)::numeric, 1),
        'students', COUNT(*) FILTER (WHERE role = 'Student'),
        'researchers', COUNT(*) FILTER (WHERE role = 'Faculty / Researcher'),
        'professionals', COUNT(*) FILTER (WHERE role = 'Professional'),
        'learning_yes', COUNT(*) FILTER (WHERE learning_tool_value = 'yes'),
        'learning_maybe', COUNT(*) FILTER (WHERE learning_tool_value = 'maybe'),
        'interested_yes', COUNT(*) FILTER (WHERE interest_in_trying = 'yes')
    ) INTO result
    FROM validation_entries;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
