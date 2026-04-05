const menuToggle = document.getElementById("menuToggle");
const navLinks = document.querySelector(".nav-links");

if (menuToggle && navLinks) {
  menuToggle.addEventListener("click", () => {
    navLinks.classList.toggle("show");
  });

  navLinks.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      navLinks.classList.remove("show");
    });
  });
}

document.querySelectorAll("[data-model-switcher]").forEach((switcher) => {
  const buttons = switcher.querySelectorAll("[data-model-target]");
  const panels = switcher.querySelectorAll("[data-model-panel]");

  if (!buttons.length || !panels.length) return;

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const target = button.getAttribute("data-model-target");
      if (!target) return;

      buttons.forEach((btn) => {
        btn.classList.toggle("active", btn === button);
      });

      panels.forEach((panel) => {
        const isActive = panel.getAttribute("data-model-panel") === target;
        panel.classList.toggle("active", isActive);
        panel.hidden = !isActive;
      });
    });
  });
});

document.querySelectorAll("[data-image-zoom]").forEach((zoomBox) => {
  const image = zoomBox.querySelector("[data-zoomable-image]");
  if (!image) return;

  const minScale = 0.25;
  const maxScale = 3;
  const step = 0.2;
  let scale = 1;

  const applyScale = () => {
    image.style.transform = `scale(${scale})`;
  };

  zoomBox.querySelectorAll("[data-zoom-action]").forEach((button) => {
    button.addEventListener("click", () => {
      const action = button.getAttribute("data-zoom-action");
      if (action === "in") {
        scale = Math.min(maxScale, scale + step);
      } else if (action === "out") {
        scale = Math.max(minScale, scale - step);
      } else {
        scale = 1;
      }
      applyScale();
    });
  });

  applyScale();
});

document.querySelectorAll("[data-tab-group]").forEach((group) => {
  const tabs = group.querySelectorAll("[data-tab-btn]");
  const root = group.parentElement;
  if (!root || !tabs.length) return;
  const panels = root.querySelectorAll("[data-tab-panel]");

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.getAttribute("data-tab-btn");
      if (!target) return;

      tabs.forEach((otherTab) => {
        otherTab.classList.toggle("active", otherTab === tab);
      });

      panels.forEach((panel) => {
        const isActive = panel.getAttribute("data-tab-panel") === target;
        panel.classList.toggle("active", isActive);
        panel.hidden = !isActive;
      });
    });
  });
});

document
  .querySelectorAll('.compare-panel[data-tab-panel="performance"] .compare-table')
  .forEach((table) => {
    const headerCells = [...table.querySelectorAll("thead th")];
    const bodyRows = [...table.querySelectorAll("tbody tr")];
    if (!headerCells.length || !bodyRows.length) return;

    const metricColumnIndexes = headerCells
      .map((th, idx) => ({ idx, label: th.textContent?.toLowerCase() || "" }))
      .filter(
        ({ label }) =>
          (label.includes("accuracy") ||
            label.includes("precision") ||
            label.includes("recall") ||
            label.includes("f1")) &&
          !label.includes("loss")
      )
      .map(({ idx }) => idx);

    metricColumnIndexes.forEach((colIdx) => {
      const numericCells = bodyRows
        .map((row) => row.querySelectorAll("td")[colIdx])
        .filter(Boolean)
        .map((cell) => {
          const raw = (cell.textContent || "").replace(/,/g, "").trim();
          const value = Number(raw);
          return { cell, value };
        })
        .filter(({ value }) => Number.isFinite(value));

      if (!numericCells.length) return;

      const maxValue = Math.max(...numericCells.map(({ value }) => value));
      const eps = 1e-12;

      numericCells.forEach(({ cell, value }) => {
        if (Math.abs(value - maxValue) > eps) return;
        cell.classList.add("best-score");
        if (!cell.querySelector(".score-pill")) {
          const original = cell.textContent?.trim() || "";
          cell.innerHTML = `<span class="score-pill">${original}</span>`;
        }
      });
    });
  });

document.querySelectorAll("[data-accordion]").forEach((accordion) => {
  const items = accordion.querySelectorAll(".accordion-item");

  items.forEach((item) => {
    const button = item.querySelector("[data-accordion-toggle]");
    const content = item.querySelector(".accordion-content");
    if (!button || !content) return;

    button.addEventListener("click", () => {
      const isOpen = item.classList.contains("is-open");
      item.classList.toggle("is-open", !isOpen);
      button.setAttribute("aria-expanded", String(!isOpen));
      content.hidden = isOpen;
    });
  });
});

document.querySelectorAll("[data-indoor-gallery]").forEach((gallery) => {
  const dataNode = document.getElementById("indoorGalleryData");
  if (!dataNode) return;

  let galleryData = {};
  try {
    galleryData = JSON.parse(dataNode.textContent || "{}");
  } catch {
    return;
  }

  const groups = Object.keys(galleryData);
  if (!groups.length) return;

  const groupTabs = gallery.querySelector("[data-gallery-groups]");
  const classList = gallery.querySelector("[data-gallery-class-list]");
  const imageGrid = gallery.querySelector("[data-gallery-image-grid]");
  const emptyLabel = gallery.querySelector("[data-gallery-empty]");
  const selectAllBtn = gallery.querySelector("[data-gallery-select-all]");
  const selectNoneBtn = gallery.querySelector("[data-gallery-select-none]");
  const toggleListBtn = gallery.querySelector("[data-gallery-toggle-list]");

  if (
    !groupTabs ||
    !classList ||
    !imageGrid ||
    !emptyLabel ||
    !selectAllBtn ||
    !selectNoneBtn ||
    !toggleListBtn
  ) {
    return;
  }

  const selectedByGroup = {};
  groups.forEach((group) => {
    selectedByGroup[group] = new Set();
  });

  let activeGroup = groups[0];

  function renderImages() {
    imageGrid.innerHTML = "";
    const selected = selectedByGroup[activeGroup];
    const scenes = galleryData[activeGroup] || [];
    const shown = scenes.filter((scene) => selected.has(scene.label));

    if (!shown.length) {
      emptyLabel.hidden = false;
      return;
    }

    emptyLabel.hidden = true;
    shown.forEach((scene) => {
      const card = document.createElement("figure");
      card.className = "indoor-image-card";
      card.innerHTML = `
        <img src="${scene.image}" alt="${scene.label}" />
        <figcaption>${activeGroup} | ${scene.label}</figcaption>
      `;
      imageGrid.appendChild(card);
    });
  }

  function renderClassList() {
    classList.innerHTML = "";
    const scenes = galleryData[activeGroup] || [];
    const selected = selectedByGroup[activeGroup];

    scenes.forEach((scene) => {
      const id = `scene-${activeGroup}-${scene.label}`.replace(/\s+/g, "-");
      const label = document.createElement("label");
      label.className = "indoor-class-item";
      label.setAttribute("for", id);
      label.innerHTML = `
        <input id="${id}" type="checkbox" ${selected.has(scene.label) ? "checked" : ""} />
        <span>${scene.label}</span>
      `;
      const checkbox = label.querySelector("input");
      checkbox?.addEventListener("change", () => {
        if (checkbox.checked) selected.add(scene.label);
        else selected.delete(scene.label);
        renderImages();
      });
      classList.appendChild(label);
    });

    renderImages();
  }

  function renderGroupTabs() {
    groupTabs.innerHTML = "";
    groups.forEach((group) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `indoor-group-btn${group === activeGroup ? " active" : ""}`;
      btn.textContent = group;
      btn.addEventListener("click", () => {
        activeGroup = group;
        renderGroupTabs();
        renderClassList();
      });
      groupTabs.appendChild(btn);
    });
  }

  selectAllBtn.addEventListener("click", () => {
    const selected = selectedByGroup[activeGroup];
    (galleryData[activeGroup] || []).forEach((scene) => selected.add(scene.label));
    renderClassList();
  });

  selectNoneBtn.addEventListener("click", () => {
    selectedByGroup[activeGroup].clear();
    renderClassList();
  });

  toggleListBtn.addEventListener("click", () => {
    classList.hidden = !classList.hidden;
    toggleListBtn.textContent = classList.hidden
      ? "▶ Show/Hide Class Selection"
      : "▼ Show/Hide Class Selection";
  });

  renderGroupTabs();
  renderClassList();
});

document.querySelectorAll("[data-news-gallery]").forEach((gallery) => {
  const dataNode = document.getElementById("multimodalSampleData");
  if (!dataNode) return;

  let samples = [];
  try {
    samples = JSON.parse(dataNode.textContent || "[]");
  } catch {
    return;
  }

  if (!Array.isArray(samples) || !samples.length) return;

  const sections = [...new Set(samples.map((item) => item.section).filter(Boolean))];
  if (!sections.length) return;

  const tabsRoot = gallery.querySelector("[data-news-sections]");
  const cardsRoot = gallery.querySelector("[data-news-cards]");
  if (!tabsRoot || !cardsRoot) return;

  let activeSection = sections[0];

  function renderCards() {
    cardsRoot.innerHTML = "";
    const visibleItems = samples.filter((item) => item.section === activeSection);

    visibleItems.forEach((item) => {
      const linkUrl = item.source_url || item.image_url || "#";
      const card = document.createElement("article");
      card.className = "news-card";
      card.innerHTML = `
        <a class="news-card-media-link" href="${linkUrl}" target="_blank" rel="noopener noreferrer">
          <img src="${item.image_url}" alt="${item.headline}" loading="lazy" />
        </a>
        <div class="news-card-body">
          <h3 class="news-card-headline">
            <a class="news-card-title-link" href="${linkUrl}" target="_blank" rel="noopener noreferrer">${item.headline}</a>
          </h3>
          <p class="news-card-abstract">${item.abstract}</p>
          <p class="news-card-caption">${item.caption}</p>
        </div>
      `;
      cardsRoot.appendChild(card);
    });
  }

  function renderTabs() {
    tabsRoot.innerHTML = "";
    sections.forEach((section) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `news-section-btn${section === activeSection ? " active" : ""}`;
      btn.textContent = section;
      btn.addEventListener("click", () => {
        activeSection = section;
        renderTabs();
        renderCards();
      });
      tabsRoot.appendChild(btn);
    });
  }

  renderTabs();
  renderCards();
});
