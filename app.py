from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Estructura de criterios CNA con niveles
criterios = [
{
        "dimension": "I Docencia y Resultados del Proceso Formativo",
        "criterio": "Criterio 1: Modelo educativo y diseño curricular",
        "niveles": {
            "Nivel 1": "El modelo educativo de la universidad está explícitamente formulado y es coherente con su proyecto institucional. Ambos orientan los procesos de enseñanza y aprendizaje para el logro de los perfiles de egreso de los distintos programas con ducentes a títulos y grados académicos. El diseño curricular se hace cargo del perfil de ingreso y las características de las y los estudiantes, para la progresión de suproceso formativo. La oferta académica se construye en consideración a los propósitos del proyecto institucional,las capacidades internas y las demandas del entorno. Se dispone de orientaciones, a nivel de diseño e implementación, para el desarrollo curricularde los programas.",
            "Nivel 2": "La implementación del modelo educativo es monitoreada sistemáticamente. El modelo educativo se revisa y actualiza en concordancia con las necesidades y los resultados obtenidos por las y los estudiantes.",
            "Nivel 3": "El diseño curricular contempla el ajuste y actualización de los programas, en base al modelo educativo y los resultados obtenidos en el logro del perfil de egreso, considerando el mediolaboral pertinente y la retroalimentación de las y los titulados y graduados."
        }
    },
    {
        "dimension": "I Docencia y Resultados del Proceso Formativo",
        "criterio": "Criterio 2: Procesos y resultados de enseñanza y aprendizaje",
        "niveles": {
            "Nivel 1": "La universidad emprende y desarrolla procesos formativos pertinentes a los perfiles de egreso, promoviendo una docencia centrada en el aprendizaje. Considera, además, el diagnóstico de brechas respecto del perfil de ingreso, implementando acciones de apoyo a la progresión en todos los niveles formativos y modalidades. Alrespecto, se definen indicadores y estos son monitoreados. Contempla criterios de admisión de acuerdo con su modelo educativo. Se disponen apoyos para el bienestar estudiantil, en los ámbitos administrativos, académicos y socioafectivos, los que contribuyen a la integración de las y los estudiantes, y favorecen el proceso de enseñanza y aprendizaje.",
            "Nivel 2": "La universidad define y evalúa periódicamente indicadores de progresión de las y los estudiantes y el nivel de logro del perfil de egreso. Estos indicadores se usan para diseñar e implementar estrategias de mejoramiento y apoyo para el aprendizaje de las y los estudiantes. La universidad promueve y desarrolla el compromiso estudiantil con el aprendizaje, basado en una gestión articulada y permanente.",
            "Nivel 3": "Las estrategias institucionales de monitoreo de la progresión estudiantil y del nivel de logro del perfil de egreso, contribuyen a mejorar los procesos de enseñanza y aprendizaje de todos los programas. Las estrategias de mejoramiento y apoyo para el aprendizaje evidencian su eficacia para el cierre de brechas respecto del perfil de ingreso esperado. La universidad es capaz de mostrar evidencias de que sus acciones generan la mejora continua de los indicadores de procesos y resultados de la enseñanza y aprendizaje."
        }
    },
    {
        "dimension": "I Docencia y Resultados del Proceso Formativo",
        "criterio": "Criterio 3: Cuerpo académico",
        "niveles": {
            "Nivel 1": "La dotación del cuerpo académico permite atender las necesidades formativas de toda la oferta académica. Existen políticas y normas parael ingreso y la evaluación periódicaa dicho cuerpo, para garantizar idoneidad disciplinar, profesional y apresto pedagógico. La universidad implementa actividades de desarrollo permanentes para la comprensión y ejecución del modelo educativo, y el fortalecimiento del sello institucional a las y los académicos.",
            "Nivel 2": "La universidad evalúa continuamente al cuerpo académico para el mejoramiento de indicadores relativos a su dotación, estabilidad, dedicación y desarrollo, en función de sus programas, niveles y modalidades. La universidad dispone de un sistema de apoyo permanente a la docencia. Se desarrollan sistemáticamente actividades de fortalecimiento académico.",
            "Nivel 3": "Las políticas y procedimientos institucionales, para el ingreso y el desarrollo del cuerpo académico, inciden en la mejora continua en los resultados de formación de las y los estudiantes. Consecuentemente, se evidencian mejoras en los indicadores de implementación y cumplimiento del modelo educativo."
        }
    },
    {
        "dimension": "I Docencia y Resultados del Proceso Formativo",
        "criterio": "Criterio 4: Investigación, innovación docente y mejora del proceso formativo",
        "niveles": {
            "Nivel 1": "La universidad cuenta con instancias que le permiten analizarlas prácticas docentes y los resultados formativos, para implementar cambios y mejoras en estos procesos.",
            "Nivel 2": "La universidad dispone de instancias o acciones sistemáticas para fomentar la investigación y/o innovación sobre las propias prácticas docentes, considerando los avances de las distintas disciplinas y profesiones. Utiliza dichos conocimientos para mejorar el proceso formativo y monitorear su efectividad.",
            "Nivel 3": "Los resultados de la investigación y/o innovación docente incidenen la mejora de los procesosde enseñanza y aprendizaje, y en la actualización sistemática de las políticas académicas institucionales. La institución muestra evidencias que los resultados de la investigación y/o innovación docente redundan en la generación de nuevos conocimientos o resultados de innovación. Éstos son difundidos a las comunidades disciplinares correspondientes, a nivel nacional o internacional."
        }
    },
     {
        "dimension": "II Gestión Estratégica y Recursos Institucionales",
        "criterio": "Criterio 5: Gobierno y estructura organizacional",
        "niveles": {
            "Nivel 1": "La estructura organizacional y su sistema de gobierno permiten gestionar todas las funciones institucionales. La gestión institucional responde a los propósitos institucionales y a los requerimientos del medio externo. La universidad cuenta con un plan de desarrollo institucional de mediano y largo plazo, consistente con su misión, visión, valores y propósitos institucionales.",
            "Nivel 2": "Existen mecanismos de evaluación de desempeño que se aplican a todas las autoridades, según corresponda. El plan de desarrollo institucional se revisa periódicamente, y se construye sobre la base de la evidencia disponible sobre su propio quehacer, la situación nacional de la educación superior y su entorno relevante. La toma de decisiones es oportuna y consistente con la misión, visión, valores y propósitos institucionales. La universidad evidencia cómo su plan de desarrollo institucional guía sus decisiones estratégicas para su concreción.",
            "Nivel 3": "La universidad aplica mecanismos formales para evaluar periódicamente el diseño y funcionamientode su estructura de gobierno, y realiza ajustes cuando es necesario y pertinente. La universidad alcanza los objetivos y metas establecidos en suplanificación estratégica."
        }
    },
    {
        "dimension": "II Gestión Estratégica y Recursos Institucionales",
        "criterio": "Criterio 6: Gestión y desarrollo de personas",
        "niveles": {
            "Nivel 1": "La universidad implementa mecanismos de reclutamiento, selección, inducción, desarrollo profesional, evaluación y retiro del personal en toda la institución. Los académicos, profesionales y administrativos son suficientes, en cantidad y dedicación, para atender todas las funciones de la universidad, en sus diferentes unidades académicas y sedes. El desarrollo de las personasestá alineado con la misión y propósitos institucionales, y esconsistente con la planificación estratégica.",
            "Nivel 2": "La institución evalúa la eficacia de sus políticas y mecanismos de gestión de personas, y realiza ajustes en función de sus resultados. Las y los académicos son jerarquizados y su desempeño es evaluado periódicamente a través de directrices conocidas, en las que priman la productividado los resultados de sus actividades, y el juicio de los pares. Existen oportunidades de desarrollo de carrera para las y los académicos, profesionales y administrativos.",
            "Nivel 3": "La universidad evidencia cómo sus políticas de gestión de personas potencian las capacidades de todo su personal, y cómo estas capacidades sostienen la misión, los propósitos y el desarrollode la institución."
        }
    },
    {
        "dimension": "II Gestión Estratégica y Recursos Institucionales",
        "criterio": "Criterio 7: Convivencia, equidad de género, diversidad e inclusión",
        "niveles": {
            "Nivel 1": "La universidad promueve un ambiente favorable para la convivencia y la calidad de vida en el espacio universitario. Cuenta con una política de equidad de género, y se despliegan acciones y recursos en función de ella. Cuenta con una política de diversidad e inclusión, y se despliegan acciones y recursos en función de ella.",
            "Nivel 2": "Existen acciones concretas, que se vinculan al desarrollo de una cultura de convivencia respetuosa y calidad de vida. Cuenta con mecanismos de gestión y procesos formalizados para articular y promover acciones en la línea de equidad de género, que se evidencian en acciones concretas en los diferentes estamentos. Cuenta con mecanismos de gestión y procesos formalizados para articular y promover accionesen la línea de la diversidade inclusión, que se evidencian enacciones concretas en los diferentes estamentos.",
            "Nivel 3": "Cuenta con resultados institucionales de satisfacción en la convivencia y calidad de vida, que permiten evidenciar el impacto de las estrategias implementadas. Las estrategias institucional es para la equidad de género permiten evidenciar los resultados alcanzados y proyectar su mejoramiento. Las estrategias institucionales de atención a la diversidad e inclusión permiten evidenciar los resultados alcanzados y proyectar su mejoramiento."
        }
    },
    {
        "dimension": "II Gestión Estratégica y Recursos Institucionales",
        "criterio": "Criterio 8: Gestión de recursos",
        "niveles": {
            "Nivel 1": "La universidad cuenta con una política y mecanismos formalizados para la obtención, manejo, asignación y control de los recursos económicos en función de los propósitos institucionales. Existen políticas formalizadas para la planificación, adquisición y mantención de recursos operativos. Existen instalaciones que respondena los requerimientos particulares de los distintos programas y modalidades que imparte. Para la modalidad virtual, cuenta con una infraestructura tecnológica que garantiza acceso, disponibilidad, estabilidad, usabilidad, seguridad y continuidad del servicio en la implementación de programas, conforme a las políticas definidas. Estos recursos permiten su funcionamiento actual, y a su vez, hacen viable proyectar un plan de inversiones que posibilita sustentar su proyecto institucional.",
            "Nivel 2": "Las políticas para la planificación, adquisición, mantención, actualización y desarrollo de los recursos son aplicadas de manera transversal y sistemática. Las políticas se ejecutan mediante mecanismos formales de planificación y control, considerando los requerimientos particulares de los distintos programas y modalidades. Estas políticas y mecanismos cuentan con procedimientos para su revisión, ajuste y actualización. Se verifica un uso adecuado de los recursos en conformidad con lo establecido en el plan de desarrollo institucional. La universidad muestra evidencias de estabilidad financiera para la ejecución de su plan de desarrollo institucional. Cuenta con sistemas de información que le permiten sistematizar datos provenientes de fuentes internas y externas, y apoyar la toma de decisiones institucionales. Se aplican sistemas de monitoreo de satisfacción de las y los usuarios, y se definen acciones de mejora en respuesta a los resultados.",
            "Nivel 3": "La universidad cuenta con acceso a los recursos que exige el estado del arte de las disciplinas que desarrolla, lo que se verifica de manera equivalente en las distintas sedes, funciones, áreas, niveles, jornadas y modalidades. Se evidencia una efectiva capacidad de ajuste al planificar y realizar las inversiones requeridas. La universidad gestiona las instalaciones y equipamientos en base a las políticas de vigencia tecnológica y operativa, así como del monitoreo de la satisfacción de las y los usuarios. La universidad cuenta con un patrimonio, recursos y políticas de inversión y endeudamiento, que le permiten asegurar su funcionamiento y proyección. Asimismo, le permite enfrentar contingencias y cumplir cabalmente con los compromisos asumidos en su plan de desarrollo institucional."
        }
    },
    {
        "dimension": "III Aseguramiento Interno de la Calidad",
        "criterio": "Criterio 9: Gestión y resultados del aseguramiento interno de la calidad",
        "niveles": {
            "Nivel 1": "Existe una política de aseguramiento interno de la calidad y responsables de su implementación. Ésta propende al fortalecimiento de las capacidades de autorregulación y de mejoramient continuo, que a su vez es coherente con la misión, valores y propósitos institucionales. La universidad recoge y procesa información acerca de los resultados de su desempeño, encuanto al cumplimiento de los riterios y estándares definidos en este documento, para todas las dimensiones y en sus políticas internas, y los utiliza para identificar áreas a mejorar. La información sobre el desempeño institucional es accesible para los distintos actores: directivos, facultades y otras",
            "Nivel 2": "La universidad cuenta con mecanismos formalizados y sistemas de información que le permiten gestionar internamente la calidad para avanzar en el cumplimiento de los criterios y estándares, y enriquecerlos desde su proyecto. La universidad promueve una cultura de calidad que contempla la participación y responsabilidad de todos sus estamentos.",
            "Nivel 3": "El sistema interno de aseguramiento de la calidad garantiza la capacidad de autorregulación y el mejoramiento continuo de todas las funciones institucionales. Se evidencia el compromiso decada uno de los estamentos y personas con la cultura de calidad institucional en todo su quehacer."
        }
    },
    {
        "dimension": "III Aseguramiento Interno de la Calidad",
        "criterio": "Criterio 10: Aseguramiento de la calidad de los programas formativos",
        "niveles": {
            "Nivel 1": "La muestra intencionada de programas de formación conducentes a título y grados académicos, asumen los propósitos institucionales de aseguramiento de la calidad y presentan mecanismos destinados a asegurar la equivalencia en la implementación de los procesos, que la propia institución ha definido como de aplicación transversal. La institución ejecuta procesos de evaluación interna de sus carreras y programas de acuerdo con procedimientos y criterios formalmente establecidos.",
            "Nivel 2": "Los programas de formación conducentes a título y grados académicos evaluados en la muestra intencionada presentan evidencias de equivalencia en el cumplimiento de los procesos y resultados que la propia institución ha definido como de aplicación transversal. La institución evalúa sistemáticamente los procesos de evaluación de sus carreras y programas, de acuerdo con procedimientos y criterios formalmente establecidos.",
            "Nivel 3": "Los programas de formación conducentes a título y grados académicos evaluados de la muestra intencionada presentan un alto grado de equivalencia y muestran un avance consistente con sus respectivos proyectos de desarrollo. La institución utiliza periódicamente los resultados de la evaluación para retroalimentar los programas y generar aprendizajes transferibles."
        }
    },
        {
        "dimension": "IV Vinculación con el Medio",
        "criterio": "Criterio 11: Política y gestión de la vinculación con el medio",
        "niveles": {
            "Nivel 1": "Existe una política y modelo de vinculación con el medio coherente con la misión, valores y propósitos institucionales, que declara los impactos externos e internos que se esperan de su implementación. La política define los objetivos y propósitos de la vinculación con el medio. Cuenta con recursos y medios de gestión para la planificación y ejecución de las acciones comprometidas. El modelo se sustenta en una política de naturaleza bidireccional, que ha sido diseñada colaborativamente por la universidad y los grupos relevantes de interés que ésta ha definido.",
            "Nivel 2": "La política y modelo de vinculaciónbidireccional con el mediose aplican en toda la universidad.Los ámbitos de vinculación conel medio se encuentran identificados,formalizados y se aplicansistemáticamente en las accionesrealizadas por la universidad.Los instrumentos de vinculacióncon el medio están orientados allogro de los impactos externose internos esperados, y cuentancon indicadores que permiten elseguimiento periódico de éstos.",
            "Nivel 3": "El seguimiento de los indicadores de impacto de las acciones comprometidas permite realizar ajustes a su planificación y ejecución cuando es necesario. Existen instrumentos para fomentar la contribución de las acciones de vinculación con el medio a la formación de las y los estudiantes en todos los niveles, así como su articulación con las actividades de investigación, creación y/o innovación que se realizan en la universidad."
        }
    },
    {
        "dimension": "IV Vinculación con el Medio",
        "criterio": "Criterio 12: Resultados e impacto de la vinculación con el medio",
        "niveles": {
            "Nivel 1": "Los impactos externos e internos de las acciones de vinculación con el medio son consistentes con los propósitos y metas institucionales, así como pertinentes al entorno que se ha definido como relevante por la universidad.",
            "Nivel 2": "Los impactos generados por las actividades de vinculación con el medio son valorados positivamente por la comunidad local o regional, especialmente por los grupos relevantes de interés definidos en las políticas institucionales. Los productos o resultados de las actividades de vinculación con el medio son utilizados para retroalimentar el proceso de formación de las y los estudiantes y a las otras dimensiones del quehacer institucional.",
            "Nivel 3": "Las acciones de vinculación con el medio comprometidas y ejecutadas, muestran una mejora continua en el logro de metas e indicadores de impacto interno y externo, local o nacional, y son coherentes con la política y modelo definidas para este fin. Es posible evidenciar un impacto positivo de las actividades de vinculación con el medio en el desarrollo de los procesos formativos de las y los estudiantes, y en las actividades de investigación, creación y/o innovación realizadas por la universidad."
        }
    },
    {
        "dimension": "V Investigación, Creación y/o Innovación",
        "criterio": "Criterio 13: Política y gestión de la investigación, creación y/o innovación",
        "niveles": {
            "Nivel 1": "Los propósitos asociados a la realización de actividades de investigación, creación y/o innovación se expresan en políticas, normativas y asignación de recursos, los que permiten su implementación y el desarrollo de proyectos consistentes con la misión institucional.",
            "Nivel 2": "Cuenta con los recursos y mecanismos de gestión que permiten planificar, ejecutar y monitorear las actividades de investigación, creación y/o innovación, así como evaluar sus resultados. Las políticas de investigación, creación y/o innovación consideran la ética y los criterios de calidad propios de la comunidad científica, tecnológica, disciplinaria o artística, nacional e internacional.",
            "Nivel 3": "Las políticas y los procesos de gestión institucionales para la investigación, creación y/o innovación se aplican sistemáticamente y se ajustan en función de sus resultados. Las políticas de investigación, creación y/o innovación responden al estado del arte y a los cambios en los medios disciplinarios, productivos y sociales pertinentes."
        } 
    },
    {
        "dimension": "V Investigación, Creación y/o Innovación",
        "criterio": "Criterio 14: Resultados de la investigación, creación y/o innovación",
        "niveles": {
            "Nivel 1": "La universidad cuenta con un plan de desarrollo institucional de las actividades de investigación, creación y/o innovación. Los resultados de las actividades de investigación, creación y/o innovación son consistentes con los propósitos y metas institucionales, así como pertinentes a las demandas regionales o nacionales.",
            "Nivel 2": "La universidad realiza investigaciónen algunos ámbitos de su quehacer. Los productos y resultados de las actividades de investigaciónson reconocidos por su impacto a nivel nacional o internacional. Las actividades de creación y/o innovación son difundidas o transferidas a nivel nacional o internacional. La universidad participa en redes colaborativas o posee convenios formalizados de investigación, creación y/o innovación con instituciones nacionales o internacionales. Los productos de las actividades de investigación, creación y/o innovación permiten contar con programas de postgrado acreditados en las áreas institucionales definidas en su plan de desarrollo institucional. La universidad obtiene regularmente fondos concursables abiertos y competitivos, nacionaleso internacionales, en las áreas o líneas de investigación, creación y/o innovación contenidas en su plan de desarrollo institucional.",
            "Nivel 3": "La universidad realiza investigaciónen todas las áreas de su quehacer. Los productos y resultados de las actividades de investigaciónson reconocidos por su impacto a nivel internacional y son considerados como una contribución significativa al área de estudio. Las actividades de creación y/o innovación son difundidas o transferidas a nivel internacional. La universidad muestra evidenciasde participación en redes colaborativas y posee convenios formalizados de investigación, creación y/o innovación con instituciones internacionales. Los resultados de investigación, creación y/o innovación permiten sostener programas de doctorados acreditados, en todaslas áreas del conocimiento que desarrolla la universidad. Existen mecanismos para analizar, evaluar y fomentar, de manera sistemática, la contribución de las actividades de investigación,creación y/o innovación ala formación de las y los estudiantesen todos los niveles."
        }
    }
]

# Generar embeddings para cada nivel
for c in criterios:
    c["nivel_embeddings"] = {}
    for nivel, descripcion in c["niveles"].items():
        c["nivel_embeddings"][nivel] = model.encode(descripcion, convert_to_tensor=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    resultados = []
    texto_ingresado = ""

    if request.method == 'POST':
        texto_ingresado = request.form.get('actividad', '')
        if texto_ingresado:
            texto_embedding = model.encode(texto_ingresado, convert_to_tensor=True)

            for c in criterios:
                for nivel, nivel_embedding in c["nivel_embeddings"].items():
                    similitud = util.cos_sim(texto_embedding, nivel_embedding).item()
                    resultados.append({
                        "dimension": c["dimension"],
                        "criterio": c["criterio"],
                        "nivel": nivel,
                        "porcentaje": round(similitud * 100, 2)
                    })

            resultados.sort(key=lambda x: x["porcentaje"], reverse=True)
            resultados = resultados[:10]  # Mostrar solo los 10 primeros

    return render_template('index.html', resultados=resultados, texto=texto_ingresado)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5050, debug=True)
