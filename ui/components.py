from __future__ import annotations
from typing import Optional
import streamlit as st


def section_card(title: Optional[str] = None, help: Optional[str] = None):
    """Context manager to render a card-like container for sections."""
    container = st.container()
    with container:
        if title:
            cols = st.columns([1, 0.08]) if help else None
            if help and cols:
                with cols[0]:
                    st.subheader(title)
                with cols[1]:
                    with st.popover("ⓘ"):
                        st.markdown(help)
            else:
                st.subheader(title)
        yield container


def info_tip(label: str, title: str, body_md: str):
    try:
        pop = st.popover(label)
        with pop:
            st.markdown(f"**{title}**")
            st.markdown(body_md)
    except Exception:
        with st.expander(label):
            st.markdown(f"**{title}**")
            st.markdown(body_md)


def button_bar(buttons: list[tuple[str, str]]):
    """Render a horizontal bar of buttons.
    buttons: list of (label, key) tuples
    Returns the key of the clicked button or None.
    """
    cols = st.columns(len(buttons))
    for i, (label, key) in enumerate(buttons):
        if cols[i].button(label, key=key):
            return key
    return None
