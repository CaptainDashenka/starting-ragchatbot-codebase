"""
Test data utilities for generating test scenarios and sample data.
"""

from typing import List, Dict, Any
from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


class TestDataGenerator:
    """Utility class for generating test data"""
    
    @staticmethod
    def create_sample_course(
        title: str = "Sample Course",
        instructor: str = "Test Instructor",
        course_link: str = "https://example.com/course",
        num_lessons: int = 2
    ) -> Course:
        """Create a sample course with specified number of lessons"""
        lessons = []
        for i in range(1, num_lessons + 1):
            lessons.append(Lesson(
                lesson_number=i,
                title=f"Lesson {i}: Test Topic {i}",
                lesson_link=f"https://example.com/course/lesson{i}"
            ))
        
        return Course(
            title=title,
            course_link=course_link,
            instructor=instructor,
            lessons=lessons
        )
    
    @staticmethod
    def create_course_chunks(
        course_title: str,
        lesson_count: int = 2,
        chunks_per_lesson: int = 3
    ) -> List[CourseChunk]:
        """Create sample course chunks for testing"""
        chunks = []
        chunk_index = 0
        
        for lesson_num in range(1, lesson_count + 1):
            for chunk_num in range(chunks_per_lesson):
                chunk_content = f"""
                This is chunk {chunk_num + 1} from lesson {lesson_num} of {course_title}.
                It contains educational content about advanced topics in the field.
                The content includes examples, explanations, and practical applications
                that students can learn from. This chunk simulates real course material
                with sufficient length for vector embedding testing.
                """
                
                chunks.append(CourseChunk(
                    content=chunk_content.strip(),
                    course_title=course_title,
                    lesson_number=lesson_num,
                    chunk_index=chunk_index
                ))
                chunk_index += 1
        
        return chunks
    
    @staticmethod
    def create_anthropic_courses() -> List[Course]:
        """Create test courses based on Anthropic-related topics"""
        courses = [
            Course(
                title="Building Towards Computer Use with Anthropic",
                course_link="https://www.deeplearning.ai/short-courses/building-toward-computer-use-with-anthropic/",
                instructor="Colt Steele",
                lessons=[
                    Lesson(
                        lesson_number=0,
                        title="Introduction",
                        lesson_link="https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/a6k0z/introduction"
                    ),
                    Lesson(
                        lesson_number=1,
                        title="Anthropic's Claude models",
                        lesson_link="https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/b7l1a/anthropics-claude-models"
                    ),
                    Lesson(
                        lesson_number=2,
                        title="Getting Started with the Anthropic API",
                        lesson_link="https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/c8m2b/getting-started"
                    )
                ]
            ),
            Course(
                title="Introduction to Large Language Models",
                course_link="https://example.com/llm-course",
                instructor="Dr. AI Expert",
                lessons=[
                    Lesson(
                        lesson_number=1,
                        title="What are LLMs?",
                        lesson_link="https://example.com/llm/lesson1"
                    ),
                    Lesson(
                        lesson_number=2,
                        title="Training Large Models",
                        lesson_link="https://example.com/llm/lesson2"
                    )
                ]
            )
        ]
        return courses


class QueryTestData:
    """Test data for different types of queries"""
    
    CONTENT_QUERIES = [
        "How do you use the Anthropic API?",
        "What is computer use with large language models?",
        "Explain prompt caching in Claude",
        "How do you implement tool calling?",
        "What are the benefits of multimodal requests?"
    ]
    
    COURSE_OUTLINE_QUERIES = [
        "What lessons are in the Computer Use course?",
        "Show me the course outline for Anthropic course",
        "List all lessons in the LLM course",
        "What's the structure of the building computer use course?"
    ]
    
    GENERAL_QUERIES = [
        "What is artificial intelligence?",
        "How do neural networks work?",
        "What's the difference between AI and ML?",
        "Explain machine learning basics"
    ]
    
    INVALID_QUERIES = [
        "",
        "   ",
        "askdfjklasdjfklasdjfklasdjfkl",
        "Find course about nonexistent topic xyz123",
        "Show me lesson 999 from imaginary course"
    ]


class MockResponses:
    """Mock responses for different scenarios"""
    
    @staticmethod
    def successful_search_response() -> Dict[str, Any]:
        """Mock successful ChromaDB search response"""
        return {
            'documents': [
                [
                    "This content explains how to use the Anthropic API for building applications.",
                    "Computer use with large language models enables automated task execution."
                ]
            ],
            'metadatas': [
                [
                    {
                        'course_title': 'Building Towards Computer Use with Anthropic',
                        'lesson_number': 1,
                        'chunk_index': 0
                    },
                    {
                        'course_title': 'Building Towards Computer Use with Anthropic',
                        'lesson_number': 2,
                        'chunk_index': 1
                    }
                ]
            ],
            'distances': [[0.1, 0.2]]
        }
    
    @staticmethod
    def empty_search_response() -> Dict[str, Any]:
        """Mock empty ChromaDB search response"""
        return {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
    
    @staticmethod
    def course_catalog_response() -> Dict[str, Any]:
        """Mock course catalog search response"""
        return {
            'documents': [["Building Towards Computer Use with Anthropic"]],
            'metadatas': [[{
                'title': 'Building Towards Computer Use with Anthropic',
                'instructor': 'Colt Steele',
                'course_link': 'https://www.deeplearning.ai/short-courses/building-toward-computer-use-with-anthropic/',
                'lessons_json': '[{"lesson_number": 1, "lesson_title": "Introduction", "lesson_link": "https://example.com/lesson1"}]',
                'lesson_count': 1
            }]],
            'distances': [[0.05]]
        }
    
    @staticmethod
    def anthropic_text_response() -> str:
        """Mock Anthropic API text response"""
        return "The Anthropic API allows you to integrate Claude's capabilities into your applications. You can make requests using the Python SDK or REST API."
    
    @staticmethod
    def anthropic_tool_use_response() -> Dict[str, Any]:
        """Mock Anthropic API response that includes tool use"""
        return {
            'stop_reason': 'tool_use',
            'content': [
                {
                    'type': 'tool_use',
                    'id': 'tool_call_123',
                    'name': 'search_course_content',
                    'input': {
                        'query': 'Anthropic API usage',
                        'course_name': 'Computer Use'
                    }
                }
            ]
        }


class ErrorScenarios:
    """Error scenarios for testing error handling"""
    
    CHROMA_ERRORS = [
        ConnectionError("Failed to connect to ChromaDB"),
        ValueError("Invalid collection name"),
        RuntimeError("Database is locked"),
        Exception("Unknown ChromaDB error")
    ]
    
    ANTHROPIC_ERRORS = [
        Exception("API rate limit exceeded"),
        Exception("Invalid API key"),
        Exception("Model not available"),
        Exception("Request timeout")
    ]
    
    VECTOR_STORE_ERRORS = [
        "Database connection error",
        "Collection not found",
        "Invalid search parameters",
        "Embedding model not available"
    ]
    
    SEARCH_TOOL_ERRORS = [
        "No vector store available",
        "Invalid query format",
        "Course resolution failed",
        "Search execution timeout"
    ]


# Convenience functions for creating test scenarios
def create_test_scenario(
    courses: List[Course],
    chunks: List[CourseChunk],
    query: str,
    expected_results: int = 1
) -> Dict[str, Any]:
    """Create a complete test scenario"""
    return {
        'courses': courses,
        'chunks': chunks,
        'query': query,
        'expected_results': expected_results,
        'description': f"Test scenario for query: '{query}'"
    }


def get_common_test_scenarios() -> List[Dict[str, Any]]:
    """Get a list of common test scenarios"""
    generator = TestDataGenerator()
    
    # Create test courses
    anthropic_course = generator.create_sample_course(
        title="Building Towards Computer Use with Anthropic",
        instructor="Colt Steele",
        num_lessons=3
    )
    
    llm_course = generator.create_sample_course(
        title="Introduction to Large Language Models",
        instructor="Dr. AI Expert",
        num_lessons=2
    )
    
    # Create corresponding chunks
    anthropic_chunks = generator.create_course_chunks(
        anthropic_course.title,
        lesson_count=3,
        chunks_per_lesson=2
    )
    
    llm_chunks = generator.create_course_chunks(
        llm_course.title,
        lesson_count=2,
        chunks_per_lesson=2
    )
    
    return [
        create_test_scenario(
            courses=[anthropic_course],
            chunks=anthropic_chunks,
            query="How do you use the Anthropic API?",
            expected_results=1
        ),
        create_test_scenario(
            courses=[anthropic_course, llm_course],
            chunks=anthropic_chunks + llm_chunks,
            query="What are large language models?",
            expected_results=2
        ),
        create_test_scenario(
            courses=[],
            chunks=[],
            query="Find information about nonexistent topic",
            expected_results=0
        )
    ]