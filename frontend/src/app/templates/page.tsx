"use client";

import { useI18n } from "@/contexts/i18n-context";
import {
  Search,
  MessageSquare,
  BookOpen,
  Briefcase,
  Star,
  Play,
  Heart,
  Loader2,
  Shield,
  LayoutDashboard,
  BarChart,
} from "lucide-react";
import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useRouter } from "next/navigation";
import { apiRequest } from "@/lib/api-wrapper";
import { getApiUrl } from "@/lib/utils";
import type { Template } from "@/types/template";

// Category section types
interface CategorySection {
  id: string;
  title: string;
  icon: any;
  templates: Template[];
}

export default function TemplatesPage() {
  const { t, locale } = useI18n();
  const router = useRouter();
  const [selectedCategory, setSelectedCategory] = useState("All");
  const [searchQuery, setSearchQuery] = useState("");
  const [templates, setTemplates] = useState<Template[]>([]);
  const [loading, setLoading] = useState(true);

  const categories = [
    { id: "All", label: t("templates.categoryTitles.all") },
    { id: "Featured", label: t("templates.categoryTitles.featured") },
    {
      id: "Healthcare & Fitness",
      label: t("templates.categoryTitles.healthcare_fitness"),
    },
    {
      id: "General & Productivity",
      label: t("templates.categoryTitles.general_productivity"),
    },
    {
      id: "Customer Service",
      label: t("templates.categoryTitles.customer_service"),
    },
    {
      id: "Finance, LMS & Ops",
      label: t("templates.categoryTitles.finance_lms_ops"),
    },
    { id: "Security", label: t("templates.categoryTitles.security") },
  ];

  // Category display configuration
  const categoryConfig: Record<string, { title: string; icon: any }> = {
    Featured: {
      title: t("templates.categoryTitles.featured"),
      icon: Star,
    },
    "Healthcare & Fitness": {
      title: t("templates.categoryTitles.healthcare_fitness"),
      icon: Heart,
    },
    "General & Productivity": {
      title: t("templates.categoryTitles.general_productivity"),
      icon: LayoutDashboard,
    },
    "Customer Service": {
      title: t("templates.categoryTitles.customer_service"),
      icon: MessageSquare,
    },
    "Finance, LMS & Ops": {
      title: t("templates.categoryTitles.finance_lms_ops"),
      icon: BarChart,
    },
    Security: {
      title: t("templates.categoryTitles.security"),
      icon: Shield,
    },
  };

  // Fetch templates from API
  useEffect(() => {
    const fetchTemplates = async () => {
      try {
        setLoading(true);
        const response = await apiRequest(
          `${getApiUrl()}/api/templates/?lang=${locale}`
        );
        if (response.ok) {
          const data = await response.json();
          setTemplates(data);
        }
      } catch (error) {
        console.error("Failed to fetch templates:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchTemplates();
  }, [locale]);

  // Get icon and color for template
  const getTemplateStyle = (category: string) => {
    const styles: Record<
      string,
      { icon: any; color: string; bgColor: string }
    > = {
      Support: {
        icon: MessageSquare,
        color: "text-blue-500",
        bgColor: "bg-blue-500/10",
      },
      Sales: {
        icon: Briefcase,
        color: "text-green-500",
        bgColor: "bg-green-500/10",
      },
      "Data & Dev": {
        icon: Play,
        color: "text-purple-500",
        bgColor: "bg-purple-500/10",
      },
      Marketing: {
        icon: Star,
        color: "text-orange-500",
        bgColor: "bg-orange-500/10",
      },
      "Healthcare & Fitness": {
        icon: Heart,
        color: "text-rose-500",
        bgColor: "bg-rose-500/10",
      },
      "General & Productivity": {
        icon: LayoutDashboard,
        color: "text-teal-500",
        bgColor: "bg-teal-500/10",
      },
      "Customer Service": {
        icon: MessageSquare,
        color: "text-indigo-500",
        bgColor: "bg-indigo-500/10",
      },
      "Finance, LMS & Ops": {
        icon: BarChart,
        color: "text-yellow-500",
        bgColor: "bg-yellow-500/10",
      },
      Security: {
        icon: Shield,
        color: "text-slate-500",
        bgColor: "bg-slate-500/10",
      },
      default: {
        icon: BookOpen,
        color: "text-gray-500",
        bgColor: "bg-gray-500/10",
      },
    };

    return (
      styles[category] ||
      styles[Object.keys(styles).find((k) => category.includes(k)) || ""] ||
      styles.default
    );
  };

  // Group templates by category
  const groupTemplatesByCategory = (
    templatesList: Template[],
    includeFeaturedSection: boolean,
    includeCategorySections: boolean,
  ): CategorySection[] => {
    const grouped: Record<string, Template[]> = {};

    templatesList.forEach((template) => {
      if (includeFeaturedSection && template.featured) {
        if (!grouped.Featured) {
          grouped.Featured = [];
        }
        grouped.Featured.push(template);
      }
      if (!includeCategorySections) {
        return;
      }
      if (!grouped[template.category]) {
        grouped[template.category] = [];
      }
      grouped[template.category].push(template);
    });

    return Object.entries(grouped)
      .map(([category, templates]) => ({
        id: category.toLowerCase().replace(/\s+/g, "-"),
        title: categoryConfig[category]?.title || category,
        icon: categoryConfig[category]?.icon || BookOpen,
        templates,
      }))
      .sort((a, b) => {
        if (a.title === t("templates.categoryTitles.featured")) {
          return -1;
        }
        if (b.title === t("templates.categoryTitles.featured")) {
          return 1;
        }
        return a.title.localeCompare(b.title);
      });
  };

  // Filter and group templates
  const filteredTemplates = templates.filter((template) => {
    const matchesCategory =
      selectedCategory === "All" ||
      (selectedCategory === "Featured" && Boolean(template.featured)) ||
      template.category === selectedCategory;
    const matchesSearch =
      template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      template.description.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  const filteredSections = groupTemplatesByCategory(
    filteredTemplates,
    true,
    selectedCategory !== "Featured",
  );

  // Handle use template
  const handleUseTemplate = async (templateId: string) => {
    // Record usage
    try {
      await apiRequest(`${getApiUrl()}/api/templates/${templateId}/use`, {
        method: "POST",
      });
    } catch (error) {
      console.error("Failed to record template usage:", error);
    }

    // Navigate to build/new page with template parameter
    router.push(`/build/new?template=${templateId}`);
  };

  // Handle like template
  const handleLikeTemplate = async (
    templateId: string,
    e: React.MouseEvent,
  ) => {
    e.stopPropagation();
    try {
      const response = await apiRequest(
        `${getApiUrl()}/api/templates/${templateId}/like`,
        { method: "POST" },
      );
      if (response.ok) {
        // Refresh templates to get updated stats
        const templatesResponse = await apiRequest(
          `${getApiUrl()}/api/templates/?lang=${locale}`,
        );
        if (templatesResponse.ok) {
          const data = await templatesResponse.json();
          setTemplates(data);
        }
      }
    } catch (error) {
      console.error("Failed to like template:", error);
    }
  };

  return (
    <div className="flex flex-col h-full bg-background/50">
      {/* Header */}
      <div className="flex justify-between items-start w-full p-8">
        <div>
          <h1 className="text-3xl font-bold mb-1">{t("templates.title")}</h1>
          <p className="text-muted-foreground">{t("templates.subtitle")}</p>
        </div>
        <div className="relative w-72">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder={t("templates.searchPlaceholder")}
            className="pl-9 bg-secondary/50 border-border/50 focus:bg-background transition-all"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      {/* Category Filter */}
      <div className="w-full px-8 pb-6 flex gap-2 overflow-x-auto scrollbar-hide">
        {categories.map((category) => (
          <button
            key={category.id}
            onClick={() => setSelectedCategory(category.id)}
            className={cn(
              "px-4 py-1.5 rounded-full text-sm font-medium transition-all whitespace-nowrap",
              selectedCategory === category.id
                ? "bg-primary text-primary-foreground shadow-sm"
                : "bg-secondary/50 text-muted-foreground hover:bg-secondary hover:text-foreground",
            )}
          >
            {category.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <div className="w-full px-8 py-8 space-y-10">
            {filteredSections.map((section) => (
              <div key={section.id} className="animate-fade-in">
                {/* Section Header */}
                <div className="flex items-center gap-2 mb-4 text-foreground/90 font-medium">
                  <section.icon className="w-5 h-5 text-purple-500" />
                  <h2>{section.title}</h2>
                </div>

                {/* Templates Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 2xl:grid-cols-5 gap-4">
                  {section.templates.map((template) => {
                    const style = getTemplateStyle(template.category);
                    const Icon = style.icon;

                    return (
                      <div
                        key={template.id}
                        onClick={() => handleUseTemplate(template.id)}
                        className="cursor-pointer shadow-md p-5 rounded-xl border border-border/40 bg-card hover:border-primary/20 hover:shadow-lg transition-all duration-300"
                      >
                        <div className="flex items-start justify-between mb-4">
                          <div
                            className={cn(
                              "w-10 h-10 rounded-lg flex items-center justify-center transition-transform group-hover:scale-110",
                              style.bgColor,
                            )}
                          >
                            <Icon className={cn("w-5 h-5", style.color)} />
                          </div>
                        </div>

                        <h3 className="font-semibold text-base mb-2 group-hover:text-primary transition-colors">
                          {template.name}
                        </h3>
                        <p className="text-sm text-muted-foreground line-clamp-2 mb-4 h-10">
                          {template.description}
                        </p>

                        <div className="flex items-center justify-between text-xs text-muted-foreground mb-4">
                          <div className="flex items-center gap-3">
                            <div className="flex items-center gap-1">
                              <Play className="w-3 h-3 fill-current" />
                              <span>{template.used_count}</span>
                            </div>
                            <button
                              onClick={(e) =>
                                handleLikeTemplate(template.id, e)
                              }
                              className="flex items-center gap-1 hover:text-pink-500 transition-colors"
                            >
                              <Heart className="w-3 h-3 fill-current" />
                              <span>{template.likes}</span>
                            </button>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}

            {filteredSections.length === 0 && !loading && (
              <div className="text-center py-20 text-muted-foreground">
                <p>{t("templates.noResults")}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
