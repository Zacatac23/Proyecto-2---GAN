"""Genera el informe PDF de 12 paginas a partir de los artefactos reales."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT=Path(__file__).resolve().parent


def read(path): return json.loads(path.read_text(encoding='utf-8'))


class Report:
    def __init__(self,path):
        path.parent.mkdir(parents=True,exist_ok=True)
        pdfmetrics.registerFont(TTFont('Arial',r'C:\Windows\Fonts\arial.ttf'))
        pdfmetrics.registerFont(TTFont('Arial-Bold',r'C:\Windows\Fonts\arialbd.ttf'))
        pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial',boldItalic='Arial-Bold')
        self.c=canvas.Canvas(str(path),pagesize=letter)
        self.c.setTitle('Proyecto 2: diseno generativo de personajes')
        self.c.setAuthor('Luis Gonzalez - 23353; Jonathan Zacarias - 231104')
        self.style=ParagraphStyle('body',fontName='Arial',fontSize=11,leading=15,alignment=TA_JUSTIFY,textColor=colors.black)
        self.small=ParagraphStyle('table',parent=self.style,fontSize=10.5,leading=13,alignment=TA_LEFT)
        self.page=0; self.y=720; self.w=468
        self.layout=[]
    def new(self,title):
        if self.page:
            self.finish_page()
        self.page+=1; self.y=720
        self.p(f'<b>{escape(title)}</b>',gap=12)
    def check(self,height):
        if self.y-height<72:
            raise RuntimeError(f'Contenido excede pagina {self.page}: y={self.y}, alto={height}')
    def p(self,text,gap=8):
        paragraph=Paragraph(text,self.style)
        _,height=paragraph.wrap(self.w,640)
        self.check(height+gap)
        paragraph.drawOn(self.c,72,self.y-height)
        self.layout.append({'page':self.page,'type':'text','bottom':self.y-height,'height':height,'text':text})
        self.y-=height+gap
    def image(self,path,maxheight=320,width=None):
        img=Image.open(path); w,h=img.size
        targetw=width or self.w
        ratio=min(targetw/w,maxheight/h)
        dw,dh=w*ratio,h*ratio
        self.check(dh+9)
        self.c.drawImage(str(path),72+(self.w-dw)/2,self.y-dh,width=dw,height=dh)
        self.layout.append({'page':self.page,'type':'image','path':str(path),'bottom':self.y-dh,'height':dh})
        self.y-=dh+9
    def table(self,headers,rows,widths):
        data=[[Paragraph(f'<b>{escape(str(v))}</b>',self.small) for v in headers]]
        data += [[Paragraph(escape(str(v)),self.small) for v in row] for row in rows]
        table=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        table.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.4,colors.black),
                                  ('VALIGN',(0,0),(-1,-1),'TOP'),
                                  ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
                                  ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        _,height=table.wrap(self.w,640)
        self.check(height+12); table.drawOn(self.c,72,self.y-height)
        self.layout.append({'page':self.page,'type':'table','bottom':self.y-height,'height':height})
        self.y-=height+12
    def sprites(self,paths,labels,columns=6,size=64):
        rows=(len(paths)+columns-1)//columns
        height=rows*(size+26)
        self.check(height+8)
        spacing=self.w/columns
        for i,path in enumerate(paths):
            x=72+spacing*(i%columns)+(spacing-size)/2
            top=self.y-(i//columns)*(size+26)
            self.c.drawImage(str(path),x,top-size,width=size,height=size)
            self.c.setFont('Arial',10.5); self.c.setFillColor(colors.black)
            self.c.drawCentredString(x+size/2,top-size-14,labels[i])
        self.y-=height+8
    def gallery(self,directory,characters):
        columns=5; size=78; height=2*140
        self.check(height+8)
        for i,row in enumerate(characters):
            center=72+self.w/columns*(i%columns+0.5)
            top=self.y-(i//columns)*140
            self.c.drawImage(str(directory/row['image']),center-size/2,top-size,width=size,height=size)
            self.c.setFont('Arial-Bold',11); self.c.drawCentredString(center,top-size-16,row['name'])
            self.c.setFont('Arial',10.5); self.c.drawCentredString(center,top-size-31,row.get('role','Criatura'))
            self.c.drawCentredString(center,top-size-46,f"Seed {row['seed']}")
        self.y-=height+8
    def neighbors(self,pairs):
        size=72; rowheight=104; height=5*rowheight
        self.check(height)
        for i,(generated,neighbor,label1,label2) in enumerate(pairs):
            base=72+(i%2)*(self.w/2)
            top=self.y-(i//2)*rowheight
            for offset,path,label in [(8,generated,label1),(118,neighbor,label2)]:
                self.c.drawImage(str(path),base+offset,top-size,width=size,height=size)
                self.c.setFont('Arial',10.5)
                self.c.drawCentredString(base+offset+size/2,top-size-14,label)
        self.y-=height
    def finish_page(self):
        self.c.setFillColor(colors.black); self.c.setFont('Arial',10)
        self.c.drawCentredString(306,42,str(self.page)); self.c.showPage()
    def save(self,path):
        self.finish_page(); self.c.save()
        (path.parent/'layout.json').write_text(json.dumps(self.layout,indent=2,ensure_ascii=False),encoding='utf-8')
        assert self.page==12,self.page


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--epoch',type=int,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'output/pdf/presentacion.pdf')
    args=parser.parse_args(); run=args.run.resolve(); epoch=args.epoch
    summary=read(ROOT/'dataset/summary.json'); rows=read(ROOT/'dataset/manifest.json')
    configs={tag:read(run/tag/'config.json') for tag in ['base','hinge','hinge_sn']}
    histories={tag:read(run/tag/'history.json') for tag in configs}
    metrics={tag:read(run/tag/f'evaluation_epoch_{epoch:04d}.json') for tag in configs}
    directory=run/'galeria'; selection=read(directory/'manifest.json'); verification=read(directory/'verification.json')
    selected=selection['model']; cfg=configs[selected]; characters=selection['characters']
    report=Report(args.output.resolve())
    taglabel={'base':'BCE','hinge':'Hinge','hinge_sn':'Hinge + SN'}
    avg={tag:{key:float(np.mean([r[key] for r in [v for v in hist if v['epoch']<=epoch][-100:]])) for key in ['D','G']} for tag,hist in histories.items()}
    dsmall={tag:sum(r['D']<.1 for r in hist) for tag,hist in histories.items()}
    f=lambda n: f'{n:.3f}'

    report.new('Universo visual y datos')
    report.p('<b>Referencia:</b> Proyecto 2, Diseño generativo de personajes',gap=2)
    report.p('<b>Entidad destinataria:</b> Universidad del Valle de Guatemala, Deep Learning 2026',gap=2)
    report.p('<b>Objeto:</b> Generación de conceptos visuales mediante una GAN',gap=2)
    report.p('<b>Tipo de documento:</b> Informe de resultados en 12 páginas',gap=2)
    report.p('<b>Autores:</b> Luis Gonzalez - 23353 y Jonathan Zacarias - 231104',gap=2)
    report.p('<b>Docente:</b> Kevin Recinos',gap=2)
    report.p('<b>Fecha:</b> 11 de octubre de 2026',gap=12)
    report.p('Nuestro universo reúne habitantes de criptas, cavernas y ruinas: aventureros, criaturas y guardianes. Buscamos pixel art de baja resolución, figuras compactas, contornos oscuros y una paleta reducida. Juzgamos los resultados por la legibilidad de su silueta y su variedad visual. Los nombres y roles de la galería son interpretaciones humanas de los conceptos generados.')
    report.p('El archivo suministrado contiene 312 sprites de 16 × 16: 132 de Tiny Dungeon, de Kenney, y 180 de Tiny Creatures, de Clint Bellanger con colaboración de Kenney. Las páginas de origen declaran licencia CC0. La revisión conserva 199 imágenes y registra la decisión de cada archivo en un manifiesto.')
    report.p('Fuentes: <link href="https://kenney.nl/assets/tiny-dungeon" color="black">kenney.nl/assets/tiny-dungeon</link> y <link href="https://opengameart.org/content/tiny-creatures" color="black">opengameart.org/content/tiny-creatures</link>.')
    sample=[r for r in rows if r['keep'] and r['split']=='train'][::13][:12]
    report.sprites([ROOT/r['image_path'] for r in sample],[r['id'].replace('creatures_','c').replace('dungeon_','d') for r in sample],columns=6,size=64)

    report.new('Depuración, separación y aumento de datos')
    report.p('La revisión visual elimina 112 piezas de escenarios, objetos, armas, equipo e interfaz de Tiny Dungeon, además de una mano aislada de Tiny Creatures. Aceptamos criaturas completas, incluidos monstruos vegetales y animales. La comprobación de contenido y su reflejo no encontró duplicados exactos entre las imágenes aceptadas.')
    report.table(['Etapa','Cantidad','Tratamiento'],[
        ['Archivos originales',312,'PNG legibles de 16 × 16'],['Sprites aceptados',199,'20 de Dungeon y 179 de Creatures'],
        ['Entrenamiento',164,'102 grupos de familias'],['Validación',35,'26 grupos reservados'],
        ['Entrenamiento aumentado',328,'Original y reflejo horizontal'],
    ],[143,65,260])
    report.p('Agrupamos variantes conocidas de color, tamaño y pose, junto con siluetas exactamente iguales incluso reflejadas. La semilla 1729 asigna aproximadamente el 20% de grupos a validación antes del aumento. Ningún grupo aparece en ambas particiones. Esta regla reduce la fuga de variantes conocidas, aunque la clasificación manual puede omitir parentescos visuales.')
    report.p('Componemos la transparencia sobre negro, centramos la caja de contenido en 16 × 16 y escalamos con NEAREST a 64 × 64. Normalizamos RGB a [-1, 1]. Eliminamos las permutaciones RGB. La validación conserva sus imágenes originales y nunca participa en los gradientes.')
    subset=[r for r in rows if r['keep'] and r['split']=='validation'][:6]
    report.p('Las siguientes imágenes pertenecen exclusivamente a validación:')
    report.sprites([ROOT/r['image_path'] for r in subset],[r['id'].replace('creatures_','c').replace('dungeon_','d') for r in subset],columns=6,size=64)

    report.new('Arquitectura y aprendizaje adversarial')
    report.p('Elegimos una DCGAN compacta porque los datos tienen estructura espacial y una resolución nativa de solo 16 × 16. Aprender esa rejilla evita dedicar capacidad a los bloques repetidos que introduce el escalado a 64 × 64. La exportación NEAREST forma parte del generador y mantiene bloques de 4 × 4 uniformes.')
    report.p(f"El generador recibe z de 100 números normales. Tres convoluciones transpuestas construyen mapas de 4 × 4 con 256 canales, 8 × 8 con 128 canales y 16 × 16 con 23 canales. Las dos primeras etapas usan BatchNorm y ReLU. La red tiene {cfg['generator_parameters']} parámetros. D promedia bloques de 4 × 4 para reducir a 16 × 16; usa convoluciones 3/64/128/1, LeakyReLU y BatchNorm. Devuelve una puntuación sin Sigmoid final y tiene {cfg['discriminator_parameters']} parámetros.")
    report.p('La salida final asigna a cada píxel uno de los 23 colores presentes en entrenamiento. El pase hacia adelante usa argmax. Para aprender a través de esa selección discreta, usamos un gradiente sustituto de softmax, denominado straight-through. En evaluación la selección es determinista. Esta aproximación introduce sesgo en el gradiente y constituye una limitación del diseño.')
    report.p('Por lote, D evalúa imágenes reales y falsas y Adam ajusta sus parámetros. Después evaluamos las falsas con D actualizado y retropropagamos hasta G; Adam actualiza solo G. Los optimizadores referencian las redes ejecutadas. El generador aprende sin etiquetas de nombres ni clases.')
    report.p(f"Usamos Adam, tasa 0.0002 en ambas redes, betas 0.5 y 0.999, lote 32, semilla 42 y {cfg['epochs']} épocas. El orden de lotes y el ruido son independientes e idénticos entre condiciones. Ejecutamos en {escape(cfg['gpu'] or 'CPU')}, Python {cfg['python']} y PyTorch {escape(cfg['torch'])}.")
    report.p('Referencias: Radford, Metz y Chintala, DCGAN (2015), <link href="https://arxiv.org/abs/1511.06434" color="black">arxiv.org/abs/1511.06434</link>. Bengio, Léonard y Courville, estimadores de gradiente (2013), <link href="https://arxiv.org/abs/1308.3432" color="black">arxiv.org/abs/1308.3432</link>.')
    report.image(run/'figures/flujo.png',maxheight=180)

    report.new('Experimento de pérdida')
    report.p('<b>Hipótesis previa:</b> Creemos que Hinge mejorará la estabilidad porque usa un margen en D y una pérdida lineal en G. Lo consideraremos útil si preserva diversidad y produce siluetas más claras con el mismo ruido fijo, sin aumentar similitud excesiva con entrenamiento.')
    report.p('BCE no saturante constituye el control. Hinge cambia exclusivamente la pérdida. Conservamos datos, arquitectura con paleta, inicialización, BatchNorm, optimizador, épocas, secuencias de lotes y ruido. Los hashes de los pesos iniciales coinciden. Las hipótesis quedaron guardadas antes de entrenar.')
    report.p('Para Hinge, D minimiza el promedio de max(0, 1 - D(real)) más max(0, 1 + D(falso)). G minimiza -promedio(D(falso)). BCE utiliza BCEWithLogitsLoss, con etiquetas reales para la pérdida de G. Las magnitudes de ambas funciones no son directamente comparables.')
    report.table(['Condición','Diversidad de 200 muestras','D media, 100 épocas hasta 1500','G media, 100 épocas hasta 1500'],
        [[taglabel[t],f(metrics[t]['diversity']),f(avg[t]['D']),f(avg[t]['G'])] for t in ['base','hinge']],[90,126,126,126])
    report.p('La figura compara ocho vectores fijos. Hinge repite figuras amarillas/verdes y su diversidad es menor que BCE; no observamos la mejora visual prevista. Ambas pérdidas muestran dominancia de D. Una diversidad mayor también puede ser ruido y no demuestra calidad por sí sola.')
    report.image(run/'figures/loss_comparison.png',maxheight=210)

    report.new('Experimento de estabilización')
    report.p('<b>Hipótesis previa:</b> Creemos que la normalización espectral mejorará porque limita la amplificación de las convoluciones de D. Lo consideraremos útil si mejora las muestras y evita dominancia prolongada de D frente al control Hinge.')
    report.p('El control usa Hinge sin normalización espectral. La variante aplica spectral_norm a las tres convoluciones de D. Conservamos BatchNorm, sesgos, arquitectura, pesos iniciales y todas las condiciones restantes. Así aislamos la normalización espectral, a diferencia del experimento histórico que también retiraba BatchNorm.')
    report.table(['Condición','Diversidad','D media a 1500','G media a 1500'],
        [[taglabel[t],f(metrics[t]['diversity']),f(avg[t]['D']),f(avg[t]['G'])] for t in ['hinge','hinge_sn']],[150,106,106,106])
    decision=read(run/'selection.json')
    report.p(escape(decision['stabilization_conclusion']))
    report.image(run/'figures/stabilization_comparison.png',maxheight=210)
    report.p('Una pérdida de D cercana a cero durante muchas épocas indica separación fácil respecto al margen. Esa señal exige revisar muestras y diversidad. No prueba por sí sola memorización ni garantiza que G produzca personajes legibles. Referencia: Miyato et al., Spectral Normalization for GANs (2018), <link href="https://arxiv.org/abs/1802.05957" color="black">arxiv.org/abs/1802.05957</link>.')

    report.new('Evolución con ruido fijo')
    report.p(f"Seguimos los mismos 16 vectores z de semilla 123 en {taglabel[selected]}. Cada evaluación recalibra una copia de G con 20 lotes de 128 vectores, semilla independiente 12345. Así no modifica el entrenamiento ni sus secuencias aleatorias. El checkpoint guarda las estadísticas usadas para generar.")
    report.image(run/'figures/fixed_evolution_selected.png',maxheight=490)
    report.p('Las columnas conservan sus vectores entre épocas. La repetición de siluetas dentro de una misma fila es una señal visual de pérdida de modos, aunque sus colores cambien.')

    report.new('Curvas de pérdida y problemas observados')
    report.p('Las curvas corresponden a corridas completas de una misma etapa. La línea continua representa D y la discontinua G. Sus escalas dependen de la función de pérdida, por lo que comparamos trayectorias dentro de cada condición y verificamos las imágenes.')
    report.image(run/'figures/all_losses.png',maxheight=280)
    report.table(['Condición','Épocas con D menor que 0.1','D media a 1500','G media a 1500'],
        [[taglabel[t],f"{dsmall[t]} de {len(histories[t])}",f(avg[t]['D']),f(avg[t]['G'])] for t in configs],[112,144,106,106])
    report.p('Las corridas RGB iniciales de 300 y 1500 épocas mantuvieron estructuras difusas y pérdidas compatibles con dominancia de D. Extender las épocas no solucionó esa limitación. Registramos entonces una nueva configuración con salida de paleta, compartida por las tres condiciones. Esa exploración cambia la arquitectura y no demuestra, por sí sola, un efecto causal aislado de la paleta.')
    report.p(escape(decision['training_conclusion']))

    report.new('Evaluación y elección del modelo')
    report.p(f"Comparamos 200 candidatos por condición, con semillas 1000 a 1199 y checkpoint de época {epoch}. Medimos RMS sobre la rejilla nativa y similitud coseno en ResNet-18 preentrenada, con entrada NEAREST de 224 × 224. Los vecinos de entrenamiento incluyen originales y sus reflejos: 328 referencias.")
    report.table(['Modelo','Diversidad','RMS a entrenamiento','RMS a validación','Coseno medio'],
        [[taglabel[t],f(metrics[t]['diversity']),f(metrics[t]['nearest_train_rms_mean']),f(metrics[t]['nearest_validation_rms_mean']),f(metrics[t]['resnet_cosine_mean'])] for t in configs],
        [86,75,105,105,97])
    report.p(f"La diversidad de los originales de entrenamiento es {f(metrics[selected]['real_train_diversity'])} y la de validación {f(metrics[selected]['real_validation_diversity'])}. El tamaño y la composición de ambos conjuntos son distintos. Sus distancias a muestras generadas no constituyen una prueba concluyente de generalización.")
    report.p(escape(decision['model_choice']))
    report.image(run/'figures/model_comparison.png',maxheight=180)
    report.p('RMS favorece fondos y alineación compartidos. ResNet-18 procede de fotografías, por lo que puede agrupar por textura y color. La validación sirve para revisar y elegir el modelo, pero no hay un tercer conjunto de prueba. Una sola semilla por condición impide afirmar significación estadística o ausencia de sobreajuste.')

    report.new('Galería final y criterio de selección')
    report.p(f"La galería usa {taglabel[selected]}, época {selection['epoch']}. Seleccionamos 10 de 200 candidatos finales: 5%. En el desarrollo se generaron {selection['development_candidate_count']} candidatos para evaluación con distintas redes y checkpoints: 10/{selection['development_candidate_count']} = {100*selection['development_selection_rate']:.2f}%. Las semillas se repiten entre condiciones. Además inspeccionamos rejillas de ruido fijo. Revisamos silueta, forma, color, fondo y vecinos.")
    report.p('Cada PNG procede directamente del generador, con conversión a 8 bits. Los nombres y roles describen interpretaciones dentro del universo, no clases aprendidas por la red. Guardamos los diez vectores z y sus semillas para regenerar los conceptos.')
    report.gallery(directory,characters)
    report.p(escape(decision['gallery_conclusion']))
    report.p(f"La comprobación reprodujo los diez personajes con error máximo {max(r['max_error'] for r in verification['characters']):.8f}, menor que 2/255. Los vectores coinciden con sus semillas. Fijamos CPU y lote 1 para exportar: una primera prueba desde GPU cambió un píxel por argmax cerca de un empate. Conservamos el registro de ese fallo y la verificación corregida.")

    report.new('Vecinos más cercanos de los diez personajes')
    report.p('Para cada personaje mostramos su vecino de entrenamiento con menor RMS, considerando originales y reflejos. Esta comparación permite inspeccionar copia y semejanza estructural. Una distancia distinta de cero no demuestra originalidad semántica.')
    report.neighbors([(directory/r['image'],run/'figures'/f"neighbor_{r['number']:02d}.png",f"Generado {r['number']:02d}",f"RMS {metrics[selected]['per_candidate'][r['candidate_index']]['rms_train']:.3f}") for r in characters])

    report.new('Reflexión final, limitaciones y reproducción')
    report.p(escape(decision['reflection']))
    report.image(run/'figures/critical_case.png',maxheight=115,width=300)
    report.p('La selección manual y la paleta favorecen coherencia visual, pero no prueban que cada resultado sea un personaje completamente nuevo. Los datos incluyen solo 164 originales de entrenamiento y contienen familias relacionadas. La red todavía puede omitir extremidades, repetir arquetipos o producir proporciones extrañas. No afirmamos ausencia de memorización a partir de una sola métrica.')
    report.p('El paquete incluye los 312 originales, selección y separación, paleta de entrenamiento, configuraciones, hipótesis previas, historiales, figuras, pesos y galería. Ejecutar generar.py --verificar comprueba la regeneración. prepare_data.py reconstruye el dataset; el notebook y README detallan las tres corridas y la evaluación.')
    report.p('Usamos asistentes de IA para revisar código, depurar datos, ejecutar verificaciones y redactar el informe. Todas las imágenes generadas provienen de las redes de este proyecto. Conservamos las corridas exploratorias y sus resultados como límites de la investigación.')
    report.p('Fundamento adversarial: Goodfellow et al., Generative Adversarial Networks (2014), <link href="https://arxiv.org/abs/1406.2661" color="black">arxiv.org/abs/1406.2661</link>. Las referencias de datos, arquitectura y estabilización aparecen junto a sus respectivos análisis.')

    report.new('Matriz de evidencias')
    report.p('La matriz vincula cada requisito con su evidencia, la conclusión que permite sostener y su principal limitación.')
    report.table(['Evidencia','Página','Conclusión','Limitación'],[
        ['Universo y fuentes','1','Estilo definido y fuentes CC0 documentadas.','Coherencia visual evaluada de forma cualitativa.'],
        ['Depuración y separación','2','199 sprites, 164/35 y familias separadas antes del aumento.','El agrupamiento puede omitir relaciones visuales.'],
        ['Arquitectura funcional','3','DCGAN con salida de paleta y exportación 64 × 64.','Gradiente sustituto aproximado y pocos datos.'],
        ['Pérdida BCE frente a Hinge','4','Comparación con una sola variable modificada.','Una semilla y pérdidas con escalas distintas.'],
        ['Normalización espectral','5','Comparación Hinge/SN conservando BatchNorm.','El resultado depende de esta arquitectura.'],
        ['Ruido fijo y curvas','6, 7','Evolución de 16 vectores y dinámica documentadas.','Muestras y pérdidas no prueban generalización.'],
        ['Evaluación y elección','8','Decisión basada en imágenes y varias medidas.','Validación usada para elegir, sin prueba independiente.'],
        ['Galería regenerable','9','10/200, vectores y semillas verificados en CPU.','Curación manual y calidad visual limitada.'],
        ['Vecinos y reflexión','10, 11','Cada personaje tiene un vecino inspeccionable.','RMS y ResNet tienen sesgos de fondo y textura.'],
    ],[100,50,154,164])
    report.p('Los manifiestos y archivos JSON contienen las cifras completas. El informe distingue las observaciones de las hipótesis y conserva las limitaciones que afectan la interpretación de los resultados.')
    report.save(args.output.resolve())
    print(f'PDF: {args.output.resolve()}, 12 paginas')


if __name__=='__main__': main()
