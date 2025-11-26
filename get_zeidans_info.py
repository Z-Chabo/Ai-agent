from langchain_core.tools import tool
from typing import List
from .agentTools.zeidans_information_array import zeidans_info

@tool
def get_zeidans_info() -> List[str]:
    """Returns a list of strings containing key information about Zeidan. Use this tool to answer any questions about Zeidan."""
    return zeidans_info